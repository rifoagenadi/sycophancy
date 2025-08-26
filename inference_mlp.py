from utils import load_model
import pickle
import torch
import pyvene as pv
import argparse
from datasets import load_dataset
from tqdm.auto import tqdm
import importlib
import utils
from utils import generate_and_decode_new_tokens, to_request, create_anthropic_batch_job

def get_probe_vectors(chosen_layers, model_id, scale, num_layers, hidden_dim, use_random_direction=None):
    target_layers = [int(layer) for layer in chosen_layers]

    probes = {key: torch.zeros(hidden_dim) for key in range(num_layers)}
    for layer in target_layers:
        current_probe = torch.load(f'linear_probe/trained_probe/{model_id}/linear_probe_mlp_{layer}.pth')['linear.weight'].squeeze()
        if use_random_direction:
            current_probe = torch.normal(mean=current_probe.mean(), std=current_probe.std(), size=current_probe.shape)
        current_std = torch.std(current_probe)
        current_probe = scale * current_std * current_probe
        probes[layer] = current_probe
    return probes

def main():
    parser = argparse.ArgumentParser(description='Inference script with arguments')
    parser.add_argument('--model_id', type=str, default='gemma-3', help='Model ID to use')
    parser.add_argument('--chosen_layer', type=int, default=1, help='Number of top heads to use')
    parser.add_argument('--scale', type=float, default=-20, help='Scale factor for probe vectors')
    parser.add_argument('--api_key', type=str, default="", 
                        help='API key for Anthropic')
    
    args = parser.parse_args()
    
    model_id = args.model_id
    chosen_layer = args.chosen_layer
    scale = args.scale
    api_key = args.api_key
    
    print(f"Loading model: {model_id}")
    model, processor = load_model(model_id)
    model.eval()
    
    print(f"Loading accuracies for model: {model_id}")
    accuracies = pickle.load(open(f'linear_probe/trained_probe/{model_id}/accuracies_dict_mlp.pkl', 'rb'))
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
    
    print(f"Getting {chosen_layer} probes")
    chosen_layers = [chosen_layer]
    print(f"Top layers selected: {chosen_layers}")
    
    print(f"Creating probe vectors with scale {scale}")
    linear_probes = get_probe_vectors(chosen_layers, model_id, scale=scale, 
                                      num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM,
                                      use_random_direction=False
                                      )
    
    print("Setting up intervention components")
    if model_id == 'gemma-3':
        target_components = [{
                "component": f"language_model.model.layers[{i}].mlp.down_proj.input",
                "intervention": pv.AdditionIntervention(
                    source_representation=linear_probes[i].to("cuda")
                )
            } for i in range(NUM_LAYERS) if torch.count_nonzero(linear_probes[i])]
    else:
        target_components = [{
                "component": f"model.layers[{i}].mlp.down_proj.input",
                "intervention": pv.AdditionIntervention(
                    source_representation=linear_probes[i].to("cuda")
                )
            } for i in range(NUM_LAYERS) if torch.count_nonzero(linear_probes[i])]
    
    print("Creating interventable model")
    pv_model = pv.IntervenableModel(target_components, model=model)
    
    print("Loading TruthfulQA dataset")
    ds = load_dataset("truthfulqa/truthful_qa", "generation")
    questions_test = ds['validation']['question'][int(0.80*len(ds['validation'])):]
    correct_answers_test = ds['validation']['correct_answers'][int(0.80*len(ds['validation'])):]
    
    print("Generating answers for test questions")
    initial_answer = []
    final_answer = []
    for question in tqdm(questions_test):
        res_1, res_2 = generate_and_decode_new_tokens(question, pv_model, processor, model_id)
        initial_answer.append(res_1)
        final_answer.append(res_2)

    print("saving model responses")
    import pandas as pd
    pd.DataFrame({'initial_answer': initial_answer, 'final_answer': final_answer}).to_csv(f"predictions/answers_{chosen_layer}_{scale}_mlp.csv")

    
    print("Creating batch job for initial answers")
    requests = [to_request(f'truthfulqa_{model_id}_initial_iti_mlp-{i}', q, a, p) 
                   for i, (q, a, p) in enumerate(zip(questions_test, correct_answers_test, initial_answer))]
    initial_answer_path =  f"batch_job/initial_{chosen_layer}_mlp.jsonl"
    import json
    with open(initial_answer_path, 'w') as f:
        for item in requests:
            json_line = json.dumps(item)
            f.write(json_line + '\n')
    from openai import OpenAI
    client = OpenAI(api_key=args.api_key)
    batch_input_file = client.files.create(
        file=open(initial_answer_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"truthfulqa-{model_id}_initial_iti_{chosen_layer}_{scale}_mlp"
        }
    )

    print("Creating batch job for final answers")
    if 'gemma' in str(type(model)).lower():
        requests = [to_request(f'truthfulqa_{model_id}_final_iti_mlp-{i}', q, a, p) 
                   for i, (q, a, p) in enumerate(zip(questions_test, correct_answers_test, final_answer))]
    else:
        requests = [to_request(f'truthfulqa_{model_id}_final_iti_mlp-{i}', q, a, p) 
                   for i, (q, a, p) in enumerate(zip(questions_test, correct_answers_test, final_answer))]
    final_answer_path =  f"batch_job/final_{chosen_layer}_mlp.jsonl"
    with open(final_answer_path, 'w') as f:
        for item in requests:
            json_line = json.dumps(item)
            f.write(json_line + '\n')
    batch_input_file = client.files.create(
        file=open(final_answer_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"truthfulqa-{model_id}_final_iti_{chosen_layer}_{scale}_mlp"
        }
    )

if __name__ == "__main__":
    main()