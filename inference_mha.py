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

def get_probe_vectors(top_k_heads, model_folder_path, scale=5, num_layers=None, head_dim=None, num_heads=None, use_random_direction=None, direction_type='probe_weight'):
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
                current_probe = torch.load(f'linear_probe/trained_probe/{model_folder_path}/linear_probe_{layer}_{head}.pth')['linear.weight'].squeeze()
            elif direction_type == 'mean_mass':
                current_probe = torch.load(f'linear_probe/mean_direction/{model_folder_path}/linear_probe_{layer}_{head}.pth').squeeze()
            if use_random_direction:
                current_probe = torch.normal(mean=current_probe.mean(), std=current_probe.std(), size=current_probe.shape)
            current_probe = current_probe / torch.norm(current_probe, p=2)
            current_std = torch.std(current_probe)
            current_probe = scale * current_std * current_probe
            # current_probe = scale * current_probe
            probes[layer][head*head_dim:head_dim*(head+1)] = current_probe
    return probes

def main():
    parser = argparse.ArgumentParser(description='Inference script with arguments')
    parser.add_argument('--model_id', type=str, default='gemma-3', help='Model ID to use')
    parser.add_argument('--k_heads', type=int, default=16, help='Number of top heads to use')
    parser.add_argument('--scale', type=float, default=-20, help='Scale factor for probe vectors')
    parser.add_argument('--direction_type', type=str, default='probe_weight', help='Direction type for probe vectors')
    
    args = parser.parse_args()
    
    model_id = args.model_id
    k_heads = args.k_heads
    scale = args.scale
    
    print(f"Loading model: {model_id}")
    model, processor = load_model(model_id)
    model.eval()
    
    if model_id == 'gemma-3':
        model_folder_path = 'gemma-3-4b-it' 
    elif model_id == 'qwen-3':
        model_folder_path = 'Qwen3-4B-Instruct-2507'
    elif model_id == 'llama-3.2':
        model_folder_path = 'Llama-3.2-3B-Instruct'
    else:
        raise('model_id not supported')
    print(f"Loading probe from from {model_folder_path}")
    accuracies = pickle.load(open(f'linear_probe/trained_probe/{model_folder_path}/accuracies_dict_mha.pkl', 'rb'))
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
    linear_probes = get_probe_vectors(top_k_heads, model_folder_path, scale=scale, 
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
    pd.DataFrame({'question': questions_test, 'correct_answer': correct_answers_test,'initial_answer': initial_answer, 'final_answer': final_answer}).to_csv(f"predictions/{model_id}_answers_{k_heads}_{scale}_mha.csv")

if __name__ == "__main__":
    main()