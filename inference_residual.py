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

def get_top_layers(accuracy_dict, k=1):
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

def get_probe_vectors(top_k_layers, model_id, scale, num_layers, hidden_dim, use_random_direction=None):
    target_layers = [int(layer) for layer in top_k_layers]

    probes = {key: torch.zeros(hidden_dim) for key in range(num_layers)}
    for layer in target_layers:
        current_probe = torch.load(f'linear_probe/trained_probe/{model_id}/linear_probe_residual_{layer}.pth')['linear.weight'].squeeze()
        if use_random_direction:
            current_probe = torch.normal(mean=current_probe.mean(), std=current_probe.std(), size=current_probe.shape)
        current_std = torch.std(current_probe)
        current_probe = scale * current_std * current_probe
        probes[layer] = current_probe
    return probes

def main():
    parser = argparse.ArgumentParser(description='Inference script with arguments')
    parser.add_argument('--model_id', type=str, default='gemma-3', help='Model ID to use')
    parser.add_argument('--k_layers', type=int, default=16, help='Number of top heads to use')
    parser.add_argument('--scale', type=float, default=-20, help='Scale factor for probe vectors')
    parser.add_argument('--api_key', type=str, default="", 
                        help='API key for Anthropic')
    
    args = parser.parse_args()
    
    model_id = args.model_id
    k_layers = args.k_layers
    scale = args.scale
    api_key = args.api_key
    
    print(f"Loading model: {model_id}")
    model, processor = load_model(model_id)
    model.eval()
    
    print(f"Loading accuracies for model: {model_id}")
    accuracies = pickle.load(open(f'linear_probe/trained_probe/{model_id}/accuracies_dict_residual.pkl', 'rb'))
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
    
    print(f"Getting top {k_layers} heads")
    top_k_layers = get_top_layers(accuracies, k_layers)
    print(f"Top layers selected: {top_k_layers}")
    
    print(f"Creating probe vectors with scale {scale}")
    linear_probes = get_probe_vectors(top_k_layers, model_id, scale=scale, 
                                      num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM,
                                    #   use_random_direction=True
                                      )
    
    print("Setting up intervention components")
    if model_id == 'gemma-3':
        target_components = [{
                "component": f"language_model.model.layers[{i}].output",
                "intervention": pv.AdditionIntervention(
                    source_representation=linear_probes[i].to("cuda")
                )
                # "intervention": pv.ZeroIntervention
            } for i in range(NUM_LAYERS) if torch.count_nonzero(linear_probes[i])]
    else:
        target_components = [{
                "component": f"model.layers[{i}].output",
                "intervention": pv.AdditionIntervention(
                    source_representation=linear_probes[i].to("cuda")
                )
                # "intervention": pv.ZeroIntervention
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
    pd.DataFrame({'initial_answer': initial_answer, 'final_answer': final_answer}).to_csv(f"truthfulqa-{model_id}_answers_iti_{k_layers}_{scale}_residual.csv")

    print("Creating batch job for initial answers")
    requests = [to_request(f'truthfulqa_{model_id}_initial_iti_residual-{i}', q, a, p) 
                   for i, (q, a, p) in enumerate(zip(questions_test, correct_answers_test, initial_answer))]
    
    # create_anthropic_batch_job(requests, api_key=api_key)
    import json
    with open(f"truthfulqa-{model_id}_initial_iti_{k_layers}_{scale}_residual.jsonl", 'w') as f:
        for item in requests:
            json_line = json.dumps(item)
            f.write(json_line + '\n')
    from openai import OpenAI
    client = OpenAI(api_key="")
    batch_input_file = client.files.create(
        file=open(f"truthfulqa-{model_id}_initial_iti_{k_layers}_{scale}_residual.jsonl", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"truthfulqa-{model_id}_initial_iti_{k_layers}_{scale}_residual"
        }
    )

    print("Creating batch job for final answers")
    if 'gemma' in str(type(model)).lower():
        requests = [to_request(f'truthfulqa_{model_id}_final_iti_residual-{i}', q, a, p) 
                   for i, (q, a, p) in enumerate(zip(questions_test, correct_answers_test, final_answer))]
    else:
        requests = [to_request(f'truthfulqa_{model_id}_final_iti_residual-{i}', q, a, p) 
                   for i, (q, a, p) in enumerate(zip(questions_test, correct_answers_test, final_answer))]
    with open(f"truthfulqa-{model_id}_final_iti_{k_layers}_{scale}_residual.jsonl", 'w') as f:
        for item in requests:
            json_line = json.dumps(item)
            f.write(json_line + '\n')
    # create_anthropic_batch_job(requests, api_key=api_key)
    batch_input_file = client.files.create(
        file=open(f"truthfulqa-{model_id}_final_iti_{k_layers}_{scale}_residual.jsonl", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"truthfulqa-{model_id}_final_iti_{k_layers}_{scale}_residual"
        }
    )

if __name__ == "__main__":
    main()