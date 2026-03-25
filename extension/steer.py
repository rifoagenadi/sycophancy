import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyvene as pv
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from dialogue import DISAGREEMENT_TYPES, build_disagreement
from extract_activations import MODELS, load_model, get_model_config


SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Direction ablation intervention
# ---------------------------------------------------------------------------

class DirectionAblationIntervention(pv.SourcelessIntervention, pv.LocalistRepresentationIntervention):
    """Remove the component of activations along one or more unit directions.

    Stores a ``directions`` buffer of shape ``[k, dim]`` (k unit vectors).
    Forward computes: ``x' = x - (x @ dT) @ d`` which projects out all
    directions simultaneously.  When directions are orthogonal (e.g. different
    heads embedded into the full concatenated-head space) the projection is
    exact.
    """

    def __init__(self, directions: torch.Tensor, **kwargs):
        # pyvene base classes expect an embed_dim; we don't use it but must
        # satisfy the constructor.
        super().__init__(embed_dim=directions.shape[-1], **kwargs)
        self.register_buffer("directions", directions)  # [k, dim]

    def forward(self, base, source=None, subspaces=None, **kwargs):
        # base: [batch, seq, dim]  directions: [k, dim]
        dirs = self.directions.to(base.dtype)
        proj = (base @ dirs.T) @ dirs
        return base - proj


# ---------------------------------------------------------------------------
# Probe loading — steering vectors
# ---------------------------------------------------------------------------

def get_top_k_keys(accuracies, k):
    """Return the top-k component keys by accuracy (descending)."""
    sorted_items = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    return [key for key, _ in sorted_items[:k]]


def build_steering_vectors_mha(probe_weights, top_k_keys, scale, num_layers, num_heads, head_dim):
    """Build per-layer steering vectors from MHA probe weights.

    Each selected head contributes its normalised probe direction, scaled by
    ``scale``, to the layer's full concatenated-head vector (num_heads * head_dim).
    """
    vectors = {layer: torch.zeros(num_heads * head_dim) for layer in range(num_layers)}
    for key in top_k_keys:
        layer, head = (int(x) for x in key.split("_"))
        weight = probe_weights[key]["linear.weight"].squeeze()  # [head_dim]
        direction = weight / weight.norm(p=2)
        vectors[layer][head * head_dim:(head + 1) * head_dim] = scale * direction
    return vectors


def build_steering_vectors_mlp(probe_weights, top_k_keys, scale, num_layers, hidden_size):
    """Build per-layer steering vectors from MLP probe weights."""
    vectors = {layer: torch.zeros(hidden_size) for layer in range(num_layers)}
    for key in top_k_keys:
        layer = int(key)
        weight = probe_weights[key]["linear.weight"].squeeze()  # [hidden_size]
        direction = weight / weight.norm(p=2)
        vectors[layer] = scale * direction
    return vectors


def build_steering_vectors_residual(probe_weights, top_k_keys, scale, num_layers, hidden_size):
    """Build per-layer steering vectors from residual-stream probe weights."""
    vectors = {layer: torch.zeros(hidden_size) for layer in range(num_layers)}
    for key in top_k_keys:
        layer = int(key)
        weight = probe_weights[key]["linear.weight"].squeeze()
        direction = weight / weight.norm(p=2)
        vectors[layer] = scale * direction
    return vectors


# ---------------------------------------------------------------------------
# Probe loading — ablation directions
# ---------------------------------------------------------------------------

def build_ablation_directions_mha(probe_weights, top_k_keys, num_layers, num_heads, head_dim):
    """Build per-layer ablation direction matrices from MHA probe weights.

    Returns ``{layer: tensor[k, num_heads*head_dim]}`` where each row is a
    unit direction for one head, embedded into the full concatenated-head space.
    """
    per_layer = {}
    for key in top_k_keys:
        layer, head = (int(x) for x in key.split("_"))
        weight = probe_weights[key]["linear.weight"].squeeze()  # [head_dim]
        direction = weight / weight.norm(p=2)
        full = torch.zeros(num_heads * head_dim)
        full[head * head_dim:(head + 1) * head_dim] = direction
        per_layer.setdefault(layer, []).append(full)
    return {layer: torch.stack(dirs) for layer, dirs in per_layer.items()}


def build_ablation_directions_mlp(probe_weights, top_k_keys, hidden_size):
    """Build per-layer ablation direction matrices from MLP probe weights.

    Returns ``{layer: tensor[1, hidden_size]}`` with a single unit direction.
    """
    directions = {}
    for key in top_k_keys:
        layer = int(key)
        weight = probe_weights[key]["linear.weight"].squeeze()
        direction = weight / weight.norm(p=2)
        directions[layer] = direction.unsqueeze(0)  # [1, hidden_size]
    return directions


def build_ablation_directions_residual(probe_weights, top_k_keys, hidden_size):
    """Build per-layer ablation direction matrices from residual-stream probe weights."""
    directions = {}
    for key in top_k_keys:
        layer = int(key)
        weight = probe_weights[key]["linear.weight"].squeeze()
        direction = weight / weight.norm(p=2)
        directions[layer] = direction.unsqueeze(0)  # [1, hidden_size]
    return directions


# ---------------------------------------------------------------------------
# Intervention setup
# ---------------------------------------------------------------------------

COMPONENT_TEMPLATES = {
    "mha": {
        "gemma": "model.language_model.layers[{layer}].self_attn.o_proj.input",
        "default": "model.layers[{layer}].self_attn.o_proj.input",
    },
    "mlp": {
        "gemma": "model.language_model.layers[{layer}].mlp.down_proj.input",
        "default": "model.layers[{layer}].mlp.down_proj.input",
    },
    "residual": {
        "gemma": "model.language_model.layers[{layer}].input",
        "default": "model.layers[{layer}].input",
    },
}


def create_pv_model(model, vectors, act_type, is_gemma, device, method="steer"):
    """Wrap a model with pyvene interventions on active layers.

    ``method="steer"`` uses AdditionIntervention (existing behaviour).
    ``method="ablate"`` uses DirectionAblationIntervention to project out
    the probe directions.
    """
    tpl = COMPONENT_TEMPLATES[act_type]["gemma" if is_gemma else "default"]
    components = []
    for layer, data in vectors.items():
        if method == "steer":
            if data.count_nonzero() == 0:
                continue
            intervention = pv.AdditionIntervention(
                source_representation=data.to(device),
            )
        else:  # ablate
            intervention = DirectionAblationIntervention(
                directions=data.to(device),
            )
        components.append({
            "component": tpl.format(layer=layer),
            "intervention": intervention,
        })
    if not components:
        return model
    return pv.IntervenableModel(components, model=model)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_messages(question, is_gemma):
    """Build the initial single-turn prompt."""
    if is_gemma:
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": question + " Give me your best guess and answer as concisely as possible."}]},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question + " Give me your best guess and answer as concisely as possible."},
    ]


def append_disagreement(messages, first_answer, disagreement_text, is_gemma):
    """Append assistant answer + user disagreement turn."""
    if is_gemma:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": first_answer}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": disagreement_text}]})
    else:
        messages.append({"role": "assistant", "content": first_answer})
        messages.append({"role": "user", "content": disagreement_text})
    return messages


def tokenize(messages, processor, is_gemma, device):
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if is_gemma:
        inputs = processor.tokenizer(text=text, return_tensors="pt").to(device)
    else:
        inputs = processor(text, return_tensors="pt").to(device)
    return inputs


def tokenize_batch(messages_list, processor, is_gemma, device):
    """Tokenize a list of message lists with left-padding for batched generation."""
    texts = [
        processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in messages_list
    ]
    tokenizer = processor.tokenizer if is_gemma else processor
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    # Track per-sequence input lengths (before padding on the left, these are the
    # actual token counts) so we can strip prompts from generated output.
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    return inputs, input_lengths


def generate(model, inputs, max_new_tokens, processor):
    """Generate and decode new tokens, handling both pyvene and plain models."""
    input_len = inputs["input_ids"].shape[-1]
    is_pv = "intervenable" in str(type(model)).lower()
    if is_pv:
        _, generation = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    else:
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = generation[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True)


def generate_batch(model, inputs, input_lengths, max_new_tokens, processor):
    """Generate and decode new tokens for a batch, returning a list of strings."""
    is_pv = "intervenable" in str(type(model)).lower()
    if is_pv:
        _, generation = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    else:
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    pad_len = inputs["input_ids"].shape[-1]
    answers = []
    for i, inp_len in enumerate(input_lengths):
        # With left-padding, actual tokens start at (pad_len - inp_len).
        # Generated tokens start after the full padded input.
        new_tokens = generation[i][pad_len:]
        answers.append(processor.decode(new_tokens, skip_special_tokens=True))
    return answers


def generate_two_turns(question, disagreement_text, model, processor, is_gemma, device,
                       max_new_tokens=128):
    """Run a two-turn dialogue: question -> answer1, disagreement -> answer2."""
    messages = build_messages(question, is_gemma)
    inputs = tokenize(messages, processor, is_gemma, device)
    answer1 = generate(model, inputs, max_new_tokens, processor)

    messages = append_disagreement(messages, answer1, disagreement_text, is_gemma)
    inputs = tokenize(messages, processor, is_gemma, device)
    answer2 = generate(model, inputs, max_new_tokens, processor)

    return answer1, answer2


def generate_two_turns_batched(questions, disagreement_texts, model, processor, is_gemma,
                               device, max_new_tokens=128, batch_size=8):
    """Run two-turn dialogues for all questions using batched generation."""
    n = len(questions)

    # --- Turn 1: batch all questions ---
    all_messages = [build_messages(q, is_gemma) for q in questions]
    initial_answers = []
    for start in tqdm(range(0, n, batch_size), desc="Turn 1 batches"):
        batch_msgs = all_messages[start:start + batch_size]
        inputs, input_lengths = tokenize_batch(batch_msgs, processor, is_gemma, device)
        answers = generate_batch(model, inputs, input_lengths, max_new_tokens, processor)
        initial_answers.extend(answers)

    # --- Turn 2: append disagreement and batch again ---
    for i in range(n):
        all_messages[i] = append_disagreement(
            all_messages[i], initial_answers[i], disagreement_texts[i], is_gemma,
        )
    final_answers = []
    for start in tqdm(range(0, n, batch_size), desc="Turn 2 batches"):
        batch_msgs = all_messages[start:start + batch_size]
        inputs, input_lengths = tokenize_batch(batch_msgs, processor, is_gemma, device)
        answers = generate_batch(model, inputs, input_lengths, max_new_tokens, processor)
        final_answers.extend(answers)

    return initial_answers, final_answers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Activation steering using trained probe weights.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--act-type", type=str, required=True, choices=["mha", "mlp", "residual"])
    parser.add_argument("--probe-dir", type=str, default="probe_results",
                        help="Root directory containing probe weights.")
    parser.add_argument("--probe-disagreement", type=str, default="combined",
                        help="Disagreement type for probe weights (subdirectory name).")
    parser.add_argument("--eval-disagreement", type=str, default="epistemic",
                        choices=DISAGREEMENT_TYPES,
                        help="Disagreement type used in the test dialogue's second turn.")
    parser.add_argument("--top-k", type=int, default=16,
                        help="Number of top-accuracy components to steer with.")
    parser.add_argument("--method", type=str, default="steer", choices=["steer", "ablate"],
                        help="Intervention method: 'steer' adds scaled direction, "
                             "'ablate' projects out the direction.")
    parser.add_argument("--scale", type=float, default=-5.0,
                        help="Steering scale (negative = steer away from sycophancy). "
                             "Only used with --method=steer.")
    parser.add_argument("--output-dir", type=str, default="steering_results")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples (for debugging).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation (1 = sequential).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # --- Load model ---
    print(f"Loading model: {args.model}")
    model, processor = load_model(args.model, args.device)
    cfg = get_model_config(model)

    # Configure left-padding for batched generation
    tokenizer = processor.tokenizer if cfg["is_gemma"] else processor
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    is_gemma = cfg["is_gemma"]
    print(f"  layers={cfg['num_layers']} heads={cfg['num_heads']} "
          f"hidden={cfg['hidden_size']} head_dim={cfg['head_dim']}")

    # --- Load probe weights and accuracies ---
    probe_root = Path(args.probe_dir) / args.model / args.probe_disagreement
    acc_path = probe_root / f"{args.act_type}_accuracies.json"
    weights_path = probe_root / f"{args.act_type}_probes.pt"

    assert acc_path.exists(), f"Accuracy file not found: {acc_path}"
    assert weights_path.exists(), f"Probe weights not found: {weights_path}"

    with open(acc_path) as f:
        accuracies = json.load(f)
    probe_weights = torch.load(weights_path, map_location="cpu", weights_only=True)

    top_k_keys = get_top_k_keys(accuracies, args.top_k)
    print(f"Top-{args.top_k} components: {top_k_keys}")

    # --- Build intervention vectors ---
    if args.method == "steer":
        if args.act_type == "mha":
            vectors = build_steering_vectors_mha(
                probe_weights, top_k_keys, args.scale,
                cfg["num_layers"], cfg["num_heads"], cfg["head_dim"],
            )
        elif args.act_type == "mlp":
            vectors = build_steering_vectors_mlp(
                probe_weights, top_k_keys, args.scale,
                cfg["num_layers"], cfg["hidden_size"],
            )
        else:
            vectors = build_steering_vectors_residual(
                probe_weights, top_k_keys, args.scale,
                cfg["num_layers"], cfg["hidden_size"],
            )
    else:  # ablate
        if args.act_type == "mha":
            vectors = build_ablation_directions_mha(
                probe_weights, top_k_keys,
                cfg["num_layers"], cfg["num_heads"], cfg["head_dim"],
            )
        elif args.act_type == "mlp":
            vectors = build_ablation_directions_mlp(
                probe_weights, top_k_keys, cfg["hidden_size"],
            )
        else:
            vectors = build_ablation_directions_residual(
                probe_weights, top_k_keys, cfg["hidden_size"],
            )

    # --- Create intervened model ---
    pv_model = create_pv_model(model, vectors, args.act_type, is_gemma, args.device, method=args.method)
    active_layers = len([v for v in vectors.values() if (v.count_nonzero() > 0 if v.dim() == 1 else True)])
    print(f"Method={args.method} active on {active_layers} layers"
          + (f" (scale={args.scale})" if args.method == "steer" else ""))

    # --- Load test data (held-out 25% from same split used in extraction) ---
    print("Loading TruthfulQA test split...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation")
    split_ds = ds["validation"].train_test_split(test_size=0.25, seed=args.seed)
    ds_test = split_ds["test"]

    questions = ds_test["question"]
    correct_answers = [ans for ans in ds_test["correct_answers"]]
    incorrect_answers = [ans for ans in ds_test["incorrect_answers"]]

    if args.max_samples is not None:
        questions = questions[:args.max_samples]
        correct_answers = correct_answers[:args.max_samples]
        incorrect_answers = incorrect_answers[:args.max_samples]
    print(f"  Test samples: {len(questions)}")

    # --- Build per-sample disagreement texts ---
    disagreement_texts = [
        build_disagreement(args.eval_disagreement, ia) for ia in incorrect_answers
    ]

    # --- Generate ---
    initial_answers, final_answers = generate_two_turns_batched(
        questions, disagreement_texts, pv_model, processor, is_gemma, args.device,
        batch_size=args.batch_size,
    )

    # --- Save results ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "steer":
        tag = f"{args.model}_probe-{args.probe_disagreement}_eval-{args.eval_disagreement}_{args.act_type}_k{args.top_k}_s{args.scale}"
    else:
        tag = f"{args.model}_probe-{args.probe_disagreement}_eval-{args.eval_disagreement}_{args.act_type}_k{args.top_k}_ablate"
    out_path = out_dir / f"{tag}.csv"

    df = pd.DataFrame({
        "question": questions,
        "correct_answer": correct_answers,
        "initial_answer": initial_answers,
        "final_answer": final_answers,
        "probe_disagreement": args.probe_disagreement,
        "eval_disagreement": args.eval_disagreement,
        "method": args.method,
        "act_type": args.act_type,
        "top_k": args.top_k,
    })
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} results to {out_path}")


if __name__ == "__main__":
    main()
