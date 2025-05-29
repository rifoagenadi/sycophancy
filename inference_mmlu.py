from utils import load_model
import pickle
import torch
import pyvene as pv
import argparse
from datasets import load_dataset
from tqdm.auto import tqdm
import importlib
import utils
from utils import to_request, create_anthropic_batch_job

system_prompt = "You are a helpful assistant."
def generate_and_decode_new_tokens(prompt, model, processor, model_id, max_new_tokens=256):
    """
    Generate a response to a prompt and decode only the new tokens.
    
    Args:
        prompt (str): The input prompt text
        model: The language model to use for generation
        processor: The tokenizer/processor for encoding/decoding
        max_new_tokens (int): Maximum number of new tokens to generate
        
    Returns:
        str: The decoded text from only the newly generated tokens
    """
    if model_id == 'gemma-3':
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + " Give me your best guess and answer as concisely as possible."}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt + " Give me your best guess and answer as concisely as possible."
            }
        ]    
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
    if model_id == 'gemma-3':
        inputs = processor(text=inputs, return_tensors="pt").to('cuda')
    else:
        inputs = processor(inputs, return_tensors="pt").to('cuda')

    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        if 'intervenable' in str(type(model)).lower():
            _, generation = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = generation[0][input_len:]

    
    # Decode only the new tokens
    res_1 = processor.decode(new_tokens, skip_special_tokens=True)
    return res_1

def get_top_k_keys(accuracy_dict, k=16):
    """
    Returns the top k keys from the accuracy dictionary based on highest accuracy values.
    
    Parameters:
    -----------
    accuracy_dict : dict
        Dictionary with keys in format 'layer_head' and values representing accuracies
    k : int, default=5
        Number of top keys to return
        
    Returns:
    --------
    list
        List of tuples containing (key, accuracy) for the top k accuracies
    """
    # Sort the dictionary items by value (accuracy) in descending order
    sorted_items = sorted(accuracy_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Return the top k items (or all if k is larger than dictionary size)
    return [x[0] for x in sorted_items[:min(k, len(sorted_items))]]

def get_probe_vectors(top_k_heads, model_id, scale=5, num_layers=None, head_dim=None, num_heads=None, use_random_direction=None, direction_type='probe_weight'):
    target_heads = {}
    for head_string in top_k_heads:
        layer, head = head_string.split('_')
        layer = int(layer)
        head = int(head)
        if layer in target_heads:
            target_heads[layer].append(head)
        else:
            target_heads[layer] = [head]

    probes = {key: torch.zeros(head_dim*num_heads) for key in range(num_layers)}
    for layer in target_heads:
        for head in target_heads[layer]:
            if direction_type == 'probe_weight':
                current_probe = torch.load(f'linear_probe/trained_probe/{model_id}/linear_probe_{layer}_{head}.pth')['linear.weight'].squeeze()
            elif direction_type == 'mean_mass':
                current_probe = torch.load(f'linear_probe/mean_direction/{model_id}/linear_probe_{layer}_{head}.pth').squeeze()
            if use_random_direction:
                current_probe = torch.normal(mean=current_probe.mean(), std=current_probe.std(), size=current_probe.shape)
            current_std = torch.std(current_probe)
            current_probe = scale * current_std * current_probe
            probes[layer][head*head_dim:head_dim*(head+1)] = current_probe
    return probes

def main():
    parser = argparse.ArgumentParser(description='Inference script with arguments')
    parser.add_argument('--model_id', type=str, default='gemma-3', help='Model ID to use')
    parser.add_argument('--k_heads', type=int, default=16, help='Number of top heads to use')
    parser.add_argument('--scale', type=float, default=-20, help='Scale factor for probe vectors')
    parser.add_argument('--direction_type', type=str, default='mean_mass', help='Direction type for probe vectors')

    parser.add_argument('--api_key', type=str, default="", 
                        help='API key for Anthropic')
    
    args = parser.parse_args()
    
    model_id = args.model_id
    k_heads = args.k_heads
    scale = args.scale
    api_key = args.api_key
    
    print(f"Loading model: {model_id}")
    model, processor = load_model(model_id)
    model.eval()
    
    print(f"Loading accuracies for model: {model_id}")
    accuracies = pickle.load(open(f'linear_probe/trained_probe/{model_id}/accuracies_dict.pkl', 'rb'))
    config = model.config
    
    # Set model parameters based on model type
    if 'gemma' in str(type(model)).lower():
        NUM_LAYERS = model.config.text_config.num_hidden_layers
        HIDDEN_DIM = model.config.text_config.hidden_size
        NUM_HEADS = model.config.text_config.num_attention_heads
        HEAD_DIM = model.config.text_config.head_dim
    else:
        NUM_LAYERS = model.config.num_hidden_layers
        HIDDEN_DIM = model.config.hidden_size
        NUM_HEADS = model.config.num_attention_heads
        HEAD_DIM = model.config.head_dim
    
    print(f"Getting top {k_heads} heads")
    top_k_heads = get_top_k_keys(accuracies, k_heads)
    print(f"Top heads selected: {top_k_heads}")
    
    print(f"Creating probe vectors with scale {scale}")
    linear_probes = get_probe_vectors(top_k_heads, model_id, scale=scale, 
                                      num_layers=NUM_LAYERS, head_dim=HEAD_DIM, num_heads=NUM_HEADS,
                                      use_random_direction=False,
                                      direction_type=args.direction_type
                                      )
    
    print("Setting up intervention components")
    if model_id == 'gemma-3':
        target_components = [{
                "component": f"language_model.model.layers[{i}].self_attn.o_proj.input",
                "intervention": pv.AdditionIntervention(
                    source_representation=linear_probes[i].to("cuda")
                )
                # "intervention": pv.ZeroIntervention
            } for i in range(NUM_LAYERS) if torch.count_nonzero(linear_probes[i])]
    else:
        target_components = [{
                "component": f"model.layers[{i}].self_attn.o_proj.input",
                "intervention": pv.AdditionIntervention(
                    source_representation=linear_probes[i].to("cuda")
                )
                # "intervention": pv.ZeroIntervention
            } for i in range(NUM_LAYERS) if torch.count_nonzero(linear_probes[i])]
    
    print("Creating interventable model")
    pv_model = pv.IntervenableModel(target_components, model=model)
    
    print("Loading TruthfulQA dataset")
    from datasets import load_dataset
    dataset_dict = load_dataset("cais/mmlu", "all")

    # Let's say you want to sample from the 'test' split
    split_name = 'test'
    if split_name not in dataset_dict:
        print(f"Split '{split_name}' not found. Please choose from: {list(dataset_dict.keys())}")
        exit()

    ds_split = dataset_dict[split_name]
    print(f"\nSelected split '{split_name}' with {len(ds_split)} examples.")

    # 3. Define the number of random samples you want
    n = 500  # For example, 5 random samples

    # 4. Ensure n is not larger than the dataset size
    if n > len(ds_split):
        print(f"Warning: n ({n}) is larger than the dataset size ({len(ds_split)}).")
        print(f"Sampling all {len(ds_split)} available items instead.")
        n_samples_to_take = len(ds_split)
    else:
        n_samples_to_take = n

    # 5. Shuffle the chosen split and select the first n items
    random_sample_dataset = ds_split.shuffle(seed=42).select(range(n_samples_to_take))

    questions_test = random_sample_dataset['question']
    answers_idxs = random_sample_dataset['answer']
    choices_test = random_sample_dataset['choices']
    answers_test = [x[answers_idxs[i]] for i, x in enumerate(choices_test)]

    print("Generating answers for test questions")
    predictions = []
    for question in tqdm(questions_test):
        res_1 = generate_and_decode_new_tokens(question, pv_model, processor, model_id)
        predictions.append(res_1)
    
    print("Creating batch job for initial answers")
    requests = [to_request(f'mmlu_{model_id}_intervened-{i}', q, a, p) 
                   for i, (q, a, p) in enumerate(zip(questions_test, answers_test, predictions))]
    
    # create_anthropic_batch_job(requests, api_key=api_key)
    import json
    with open(f"mmlu-{model_id}_intervened_{k_heads}_{scale}.jsonl", 'w') as f:
        for item in requests:
            json_line = json.dumps(item)
            f.write(json_line + '\n')
    from openai import OpenAI
    client = OpenAI(api_key="")
    batch_input_file = client.files.create(
        file=open(f"mmlu-{model_id}_intervened_{k_heads}_{scale}.jsonl", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"mmlu-{model_id}_intervened_{k_heads}_{scale}"
        }
    )

if __name__ == "__main__":
    main()