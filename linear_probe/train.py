# train.py
import sys
import os
current_dir = os.path.dirname(os.path.abspath(''))
sys.path.append(current_dir)

import argparse
import pickle
import importlib
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import load_model

# Assuming these utils are in the same directory or accessible via PYTHONPATH
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import linear_probe_data_utils # Needs reload if changed during session
import extract_activation # Needs reload if changed during session
from linear_probe import LinearProbe

# --- Dataset Class ---
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Training Function ---
def train_probe(model, processor, train_dataset, val_dataset, batch_size,
                learning_rate, num_epochs, device, target_component,
                activation_type, input_dim, model_id, output_dir,
                hidden_dim, head_dim): # Pass necessary dims
    """
    Trains a linear probe on top of LLM representations.
    """
    print(f"----------- Training {activation_type.upper()} Probe for Component: {target_component} --------------")

    def collate_fn(batch):
        padded_inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Determine pad_token_id based on model type
        is_gemma = 'gemma' in str(type(model)).lower()
        if is_gemma:
            pad_val = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.eos_token_id
        else:
            pad_val = processor.pad_token_id if processor.pad_token_id is not None else processor.eos_token_id

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            padded_inputs, batch_first=True, padding_value=pad_val
        )
        labels = torch.tensor(labels)
        return padded_inputs, labels

    # Initialize the linear probe
    linear_probe = LinearProbe(input_dim).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(linear_probe.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        linear_probe.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Select representation based on activation type
            if activation_type == 'mha':
                target_layer, target_head = [int(x) for x in target_component.split('_')]
                representations = inputs[:, target_layer, target_head, :].to(torch.float32)
            elif activation_type == 'mlp' or activation_type == 'residual':
                target_layer = int(target_component)
                representations = inputs[:, target_layer, :].to(torch.float32)
            else:
                raise ValueError("Invalid activation_type specified")

            outputs = linear_probe(representations)
            loss = criterion(outputs.squeeze(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation
        linear_probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                if activation_type == 'mha':
                    target_layer, target_head = [int(x) for x in target_component.split('_')]
                    representations = inputs[:, target_layer, target_head, :].to(torch.float32)
                elif activation_type == 'mlp' or activation_type == 'residual':
                    target_layer = int(target_component)
                    representations = inputs[:, target_layer, :].to(torch.float32)
                else:
                    raise ValueError("Invalid activation_type specified")


                outputs = linear_probe(representations)
                predicted = torch.sigmoid(outputs).round()
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_acc:.2f}%")
        best_val_acc = max(best_val_acc, val_acc) # Keep track of best acc

    # Save the trained linear probe
    save_dir = os.path.join(output_dir, model_id.split('/')[-1])
    os.makedirs(save_dir, exist_ok=True)
    if activation_type == 'mha':
        save_path = os.path.join(save_dir, f"linear_probe_{target_component}.pth")
    elif activation_type == 'mlp': # mlp
         save_path = os.path.join(save_dir, f"linear_probe_mlp_{target_component}.pth")
    elif activation_type == 'residual': # mlp
         save_path = os.path.join(save_dir, f"linear_probe_residual_{target_component}.pth")
    torch.save(linear_probe.state_dict(), save_path)
    print(f"Saved probe to {save_path}")
    return best_val_acc # Return best validation accuracy over epochs

# --- Plotting Functions ---
def plot_mha_heatmap(accuracies, model_id, num_layers, num_heads, output_dir):
    """Plots a heatmap for MHA accuracies."""
    print("Generating MHA heatmap...")
    rows = []
    for key, value in accuracies.items():
        layer, head = map(int, key.split('_'))
        rows.append({'layer': layer, 'head': head, 'accuracy': value})

    df = pd.DataFrame(rows)

    # Use the actual number of layers/heads from config
    accuracy_matrix = np.zeros((num_layers, num_heads))
    accuracy_matrix.fill(np.nan)

    for index, row in df.iterrows():
        if row['layer'] < num_layers and row['head'] < num_heads:
            accuracy_matrix[int(row['layer']), int(row['head'])] = row['accuracy']

    plt.figure(figsize=(14, max(10, num_layers // 2))) # Adjust size dynamically
    cmap = plt.cm.viridis
    cmap.set_bad('lightgray')
    heatmap = plt.imshow(accuracy_matrix, cmap=cmap, aspect='auto', vmin=40, vmax=max(90, df['accuracy'].max())) # Adjust color range
    plt.colorbar(heatmap, label='Accuracy (%)')

    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.title(f"Accuracy Heatmap by Layer and Head (MHA) - {model_id.split('/')[-1]}", fontsize=16)

    plt.xticks(np.arange(0, num_heads, max(1, num_heads // 16))) # Adjust ticks dynamically
    plt.yticks(np.arange(0, num_layers, max(1, num_layers // 10)))

    save_path = os.path.join(output_dir, model_id.split('/')[-1], f"{model_id.split('/')[-1]}_accuracy_heatmap_mha.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {save_path}")
    plt.close() # Close figure to prevent display in non-interactive environments

def plot_layer_line(accuracies, model_id, num_layers, output_dir, activation_type):
    """Plots a line graph for MLP accuracies."""
    print("Generating Layer-wise line plot...")
    layers = [int(k) for k in accuracies.keys()]
    accuracies_list = list(accuracies.values())

    # Ensure data is sorted by layer
    sorted_data = sorted(zip(layers, accuracies_list))
    layers_sorted, accuracies_sorted = zip(*sorted_data)


    plt.figure(figsize=(12, 6))
    plt.plot(layers_sorted, accuracies_sorted, marker='o', linestyle='-', linewidth=2, markersize=8, color='#3366cc')

    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Chance level (50%)')

    best_layer_idx = np.argmax(accuracies_sorted)
    best_layer = layers_sorted[best_layer_idx]
    best_accuracy = accuracies_sorted[best_layer_idx]
    plt.scatter(best_layer, best_accuracy, color='red', s=150, zorder=5, label=f'Best layer: {best_layer} ({best_accuracy:.2f}%)')

    plt.xlabel('Layer Number', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title(f"Accuracy by Layer ({activation_type} Output) - {model_id.split('/')[-1]}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(min(40, min(accuracies_sorted)-5), max(90, max(accuracies_sorted)+5)) # Dynamic ylim
    plt.xticks(np.arange(0, num_layers, max(1, num_layers // 14)))
    plt.legend(loc='lower right')
    plt.tight_layout()
    save_path = os.path.join(output_dir, model_id.split('/')[-1], f"{model_id.split('/')[-1]}_accuracy_line_{activation_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved line plot to {save_path}")
    plt.close() # Close figure

# --- Main Function ---
def main(args):
    # Reload utils if needed (useful during interactive development)
    importlib.reload(linear_probe_data_utils)
    importlib.reload(extract_activation)
    from linear_probe_data_utils import construct_data

    # Load Model and Processor
    print(f"Loading model: {args.model_id}...")
    model, processor = load_model(args.model_id)
    model.eval()
    model.to(args.device)

    # Get Model Config
    print("Extracting model configuration...")
    is_gemma = 'gemma' in str(type(model)).lower()
    if is_gemma:
        config = model.config.text_config
        NUM_LAYERS = config.num_hidden_layers
        HIDDEN_DIM = config.hidden_size
        NUM_HEADS = config.num_attention_heads
        HEAD_DIM = config.head_dim
        MLP_DIM = 10240 #hardcoded
    else: # hardcoded for llama 3.2
        config = model.config
        NUM_LAYERS = config.num_hidden_layers
        HIDDEN_DIM = config.hidden_size
        NUM_HEADS = config.num_attention_heads
        HEAD_DIM = config.head_dim
        MLP_DIM = 8192
        if HEAD_DIM is None and args.activation_type == 'mha':
             raise ValueError("Cannot determine HEAD_DIM for MHA on this model architecture.")

    print(f"Model Config: Layers={NUM_LAYERS}, Heads={NUM_HEADS}, HiddenDim={HIDDEN_DIM}, HeadDim={HEAD_DIM}")


    # Load and Prepare Data
    print("Loading and preparing TruthfulQA data...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation")
    ds_train_val = ds['validation'] # Use the full validation set for train/val split
    ds_train_split = ds_train_val[:int(0.8*len(ds_train_val))] # 80% for training activation extraction/probe training
    # ds_test_split = ds_train_val[int(0.8*len(ds_train_val)):] # 20% held out? The notebooks used same split for extraction and training

    chats, labels = construct_data(ds_train_split, model=args.model_id.split('-')[0]) # Simple model name

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    numerical_labels = torch.tensor(labels)

    print("Applying chat template and tokenizing...")
    chats_templated = processor.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)
    tokenized_data = [
        processor(text=chat, return_tensors="pt")["input_ids"].squeeze()
        for chat in tqdm(chats_templated, desc="Tokenizing")
    ]

    # Split data for probe training
    print("Splitting data for probe training...")
    train_tok, val_tok, train_labels, val_labels = train_test_split(
        tokenized_data, numerical_labels, test_size=0.2, 
        random_state=3407
    )

    # Extract Activations
    print(f"Extracting {args.activation_type.upper()} activations...")
    if args.activation_type == 'mha':
        extract_fn = extract_activation.extract_mha_activation
    elif args.activation_type == 'mlp':
        extract_fn = extract_activation.extract_mlp_activation
    elif args.activation_type == 'residual':
        extract_fn = extract_activation.extract_residual_activation
    else:
        raise ValueError(f"Unsupported activation_type: {args.activation_type}. Choose 'mha', 'mlp', or 'residual'.")

    train_activation_list = []
    for datum in tqdm(train_tok, total=len(train_tok), desc="Extracting Train Activations"):
        act_tensor = extract_fn(model, processor, datum.to(args.device)) # Move datum to device
        train_activation_list.append(act_tensor.cpu()) # Move back to CPU for storage

    val_activation_list = []
    for datum in tqdm(val_tok, total=len(val_tok), desc="Extracting Val Activations"):
        act_tensor = extract_fn(model, processor, datum.to(args.device)) # Move datum to device
        val_activation_list.append(act_tensor.cpu()) # Move back to CPU for storage

    # Create Datasets for Probe Training
    train_dataset = MyDataset(train_activation_list, train_labels)
    val_dataset = MyDataset(val_activation_list, val_labels)

    # Train Probes
    print("Starting probe training...")
    accuracies = {}
    if args.activation_type == 'mha':
        probe_input_dim = HEAD_DIM
    elif args.activation_type == 'mlp':
        probe_input_dim = MLP_DIM
    elif args.activation_type == 'residual':
        probe_input_dim = HIDDEN_DIM

    if args.activation_type == 'mha':
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                target_component = f"{layer}_{head}"
                current_acc = train_probe(model, processor, train_dataset, val_dataset,
                                          args.batch_size, args.lr, args.epochs, args.device,
                                          target_component, args.activation_type, probe_input_dim,
                                          args.model_id, args.output_dir, HIDDEN_DIM, HEAD_DIM)
                accuracies[target_component] = current_acc
    elif args.activation_type == 'mlp' or args.activation_type == 'residual':
        for layer in range(NUM_LAYERS):
            target_component = f"{layer}"
            current_acc = train_probe(model, processor, train_dataset, val_dataset,
                                      args.batch_size, args.lr, args.epochs, args.device,
                                      target_component, args.activation_type, probe_input_dim,
                                      args.model_id, args.output_dir, HIDDEN_DIM, HEAD_DIM)
            accuracies[target_component] = current_acc
    else:
        raise ValueError("Invalid activation_type specified")

    # Save Accuracies
    output_subdir = os.path.join(args.output_dir, args.model_id.split('/')[-1])
    os.makedirs(output_subdir, exist_ok=True)
    acc_filename = f"accuracies_dict_{args.activation_type}.pkl"
    acc_path = os.path.join(output_subdir, acc_filename)
    print(f"Saving accuracies to {acc_path}")
    with open(acc_path, 'wb') as f:
        pickle.dump(accuracies, f)

    # Plot Results
    if args.activation_type == 'mha':
        plot_mha_heatmap(accuracies, args.model_id, NUM_LAYERS, NUM_HEADS, args.output_dir)
    elif args.activation_type == 'mlp' or args.activation_type == 'residual':
        plot_layer_line(accuracies, args.model_id, NUM_LAYERS, args.output_dir, activation_type=args.activation_type)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train linear probes on LLM activations.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID (e.g., 'gemma-3'")
    parser.add_argument("--activation_type", type=str, required=True, choices=['mha', 'mlp', 'residual'], help="Type of activation to extract and train on ('mha', 'mlp', or 'residual').")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size for probe training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs for probe training.")
    parser.add_argument("--output_dir", type=str, default="trained_probe", help="Directory to save trained probes and results.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()

    # Add current directory to path for utils etc.
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    sys.path.append(current_dir)

    main(args)