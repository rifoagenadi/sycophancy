from utils import load_model, load_test_data
import pickle
import torch
import pyvene as pv
import argparse
from datasets import load_dataset
from tqdm.auto import tqdm
import importlib
import utils
from utils import generate_and_decode_new_tokens, to_request

def get_probe_vectors(chosen_layer, model_id, scale, num_layers, hidden_dim, use_random_direction=None):
    target_layers = [int(layer) for layer in chosen_layer]
    probes = {key: torch.zeros(hidden_dim) for key in range(num_layers)}
    for layer in target_layers:
        current_probe = torch.load(f'linear_probe/trained_probe_sycophancy/{model_id}/linear_probe_mlp_{layer}.pth')['linear.weight'].squeeze()
        if use_random_direction:
            current_probe = torch.normal(mean=current_probe.mean(), std=current_probe.std(), size=current_probe.shape)
        current_probe = current_probe / torch.norm(current_probe, p=2)
        current_std = torch.load(f'linear_probe/trained_probe_sycophancy/{model_id}/std_mlp_{layer}.pt')
        current_probe = scale * current_std * current_probe
        probes[layer] = current_probe
    return probes

def main():
    parser = argparse.ArgumentParser(description='Inference script with arguments')
    parser.add_argument('--model_id', type=str, default='gemma-3', help='Model ID to use')
    parser.add_argument('--dataset_id', type=str, default='truthfulqa', help='Dataset ID to use')
    parser.add_argument('--chosen_layer', type=int, default=1, help='Layer to intervene')
    parser.add_argument('--scale', type=float, default=-5, help='Scale factor for probe vectors')
    parser.add_argument('--use_random_direction', action='store_true', help='If set, use a random direction instead of a fixed probe vector')

    
    args = parser.parse_args()

    model_id = args.model_id
    dataset_id = args.dataset_id
    chosen_layer = args.chosen_layer
    scale = args.scale
    
    print(f"Loading model: {model_id}")
    model, processor = load_model(model_id)
    model.eval()
    
    print(f"Loading accuracies for model: {model_id}")
    accuracies = pickle.load(open(f'linear_probe/trained_probe_sycophancy/{model_id}/accuracies_dict_mlp.pkl', 'rb'))
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
    chosen_layer = [chosen_layer]
    print(f"Top layers selected: {chosen_layer}")
    
    print(f"Creating probe vectors with scale {scale}")
    linear_probes = get_probe_vectors(chosen_layer, model_id, scale=scale, 
                                      num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM,
                                      use_random_direction=args.use_random_direction
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
    pv_model = pv.IntervenableModel(target_components, model=model) if scale > 0 else model
    
    print(f"Loading {args.dataset_id} dataset")
    questions_test, correct_answers_test = load_test_data(args.dataset_id)

    print("Generating answers for test questions")
    initial_answer = []
    final_answer = []
    for question in tqdm(questions_test):
        res_1, res_2 = generate_and_decode_new_tokens(question, pv_model, processor, model_id)
        initial_answer.append(res_1)
        final_answer.append(res_2)

    print("saving model responses")
    import pandas as pd
    pd.DataFrame({'question': questions_test, 'correct_answer': correct_answers_test,'initial_answer': initial_answer, 'final_answer': final_answer}).to_csv(f"predictions_sycophancy/{dataset_id}_{model_id}_answers_L{chosen_layer}_{scale}_mlp.csv")

if __name__ == "__main__":
    main()