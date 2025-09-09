import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(''))
sys.path.append(current_dir)
from utils import load_model

from datasets import load_dataset
from linear_probe_data_utils import construct_data
from tqdm.auto import tqdm
from extract_activation import extract_mha_activation, extract_residual_activation, extract_mlp_activation

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save standard deviation along the learned sycophancy direction.")
    parser.add_argument("--model_id", type=str, required=True, help="'gemma-3' or 'llama-3.2")
    parser.add_argument("--activation_type", type=str, required=True, choices=['mha', 'mlp', 'residual'], help="Type of activation to extract and train on ('mha', 'mlp', or 'residual').")

    args = parser.parse_args()
    model_id = args.model_id
    activation_type = args.activation_type

    print("Loading model")
    model, processor = load_model(model_id)

    # Load and Prepare Data
    print("Loading and preparing TruthfulQA data...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation")
    ds_train_val = ds['validation'] # Use the full validation set for train/val split
    ds_train_split = ds_train_val[:int(0.8*len(ds_train_val))] # 80% for training activation extraction/probe training
    chats, labels = construct_data(ds_train_split, model=model_id) # Simple model name

    print("Applying chat template and tokenizing...")
    chats_templated = processor.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)
    tokenized_data = [
        processor(text=chat, return_tensors="pt")["input_ids"].squeeze()
        for chat in tqdm(chats_templated, desc="Tokenizing")
    ]

    if activation_type == 'mha':
        extract_fn = extract_mha_activation
    elif activation_type == 'mlp':
        extract_fn = extract_mlp_activation
    elif activation_type == 'residual':
        extract_fn = extract_residual_activation
    else:
        raise ValueError(f"Unsupported activation_type: {activation_type}. Choose 'mha', 'mlp', or 'residual'.")

    train_activation_list = []
    for datum in tqdm(tokenized_data, total=len(tokenized_data), desc="Extracting Activations"):
        act_tensor = extract_fn(model, processor, datum.to('cuda')) # Move datum to device
        train_activation_list.append(act_tensor.cpu()) # Move back to CPU for storage
    tuning_activations = torch.stack(train_activation_list)

    NUM_LAYER = train_activation_list[0].shape[0]
    NUM_HEAD = train_activation_list[0].shape[1]
    project_val_stds = []
    print(f"Computing standard deviations: {activation_type}")
    if activation_type == 'mha':
        for layer in tqdm(range(NUM_LAYER)):
            for head in range(NUM_HEAD):
                dir = torch.load(f'trained_probe/{model_id}/linear_probe_{layer}_{head}.pth')['linear.weight'][0, :].cpu().to(torch.bfloat16)
                dir = dir/torch.norm(dir)
                activations = torch.tensor(tuning_activations[:,layer,head,:], dtype=torch.bfloat16).to("cpu")
                proj_vals = activations @ dir.T
                proj_val_std = torch.std(proj_vals)
                torch.save(proj_val_std, f'trained_probe/{model_id}/std_mha_{layer}_{head}.pt')
    elif activation_type == 'residual':
        for layer in tqdm(range(NUM_LAYER)):
            dir = torch.load(f'trained_probe/{model_id}/linear_probe_residual_{layer}.pth')['linear.weight'][0, :].cpu().to(torch.bfloat16)
            dir = dir/torch.norm(dir)
            activations = torch.tensor(tuning_activations[:,layer,:], dtype=torch.bfloat16).to("cpu")
            proj_vals = activations @ dir.T
            proj_val_std = torch.std(proj_vals)
            print(proj_val_std)
            torch.save(proj_val_std, f'trained_probe/{model_id}/std_residual_{layer}.pt')
    elif activation_type == 'mlp':
        for layer in tqdm(range(NUM_LAYER)):
            dir = torch.load(f'trained_probe/{model_id}/linear_probe_mlp_{layer}.pth')['linear.weight'][0, :].cpu().to(torch.bfloat16)
            dir = dir/torch.norm(dir)
            activations = torch.tensor(tuning_activations[:,layer,:], dtype=torch.bfloat16).to("cpu")
            proj_vals = activations @ dir.T
            proj_val_std = torch.std(proj_vals)
            print(proj_val_std)
            torch.save(proj_val_std, f'trained_probe/{model_id}/std_mlp_{layer}.pt')
    else:
        raise(f'Activation type not supported: {activation_type}')