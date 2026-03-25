import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_single_probe(train_x, train_y, val_x, val_y, input_dim,
                        lr, epochs, batch_size, device):
    """Train one linear probe and return best validation accuracy."""
    probe = LinearProbe(input_dim).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False
    )

    best_acc = 0.0
    for _ in range(epochs):
        probe.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = criterion(probe(x_batch).squeeze(-1), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = (torch.sigmoid(probe(x_batch).squeeze(-1)) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        best_acc = max(best_acc, correct / total * 100)

    return best_acc, probe


def train_probes_for_activation(activations, labels, act_type,
                                 lr, epochs, batch_size, device, seed):
    """Train linear probes for every component in an activation tensor.

    MHA:      [N, num_layers, num_heads, head_dim] -> one probe per (layer, head)
    MLP:      [N, num_layers, mlp_dim]             -> one probe per layer
    Residual: [N, num_layers, hidden_size]          -> one probe per layer

    Returns dict mapping component name -> best val accuracy.
    """
    activations = activations.float()
    labels = labels.float()
    num_layers = activations.shape[1]

    # Train/val split (75/25)
    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(indices, test_size=0.25, random_state=seed)
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    accuracies = {}
    probes = {}

    if act_type == "mha":
        num_heads = activations.shape[2]
        total = num_layers * num_heads
        pbar = tqdm(total=total, desc=f"  {act_type} probes")
        for layer in range(num_layers):
            for head in range(num_heads):
                train_x = activations[train_idx, layer, head, :]
                val_x = activations[val_idx, layer, head, :]
                input_dim = train_x.shape[-1]
                acc, probe = train_single_probe(
                    train_x, train_labels, val_x, val_labels,
                    input_dim, lr, epochs, batch_size, device
                )
                key = f"{layer}_{head}"
                accuracies[key] = acc
                probes[key] = probe.cpu().state_dict()
                pbar.update(1)
        pbar.close()
    else:  # mlp or residual
        pbar = tqdm(total=num_layers, desc=f"  {act_type} probes")
        for layer in range(num_layers):
            train_x = activations[train_idx, layer, :]
            val_x = activations[val_idx, layer, :]
            input_dim = train_x.shape[-1]
            acc, probe = train_single_probe(
                train_x, train_labels, val_x, val_labels,
                input_dim, lr, epochs, batch_size, device
            )
            key = str(layer)
            accuracies[key] = acc
            probes[key] = probe.cpu().state_dict()
            pbar.update(1)
        pbar.close()

    return accuracies, probes


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on pre-extracted activations.")
    parser.add_argument("--activations-dir", type=str, default="activations",
                        help="Root directory containing extracted activations.")
    parser.add_argument("--output-dir", type=str, default="probe_results",
                        help="Directory to save probe weights and accuracy results.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    act_types = ["mha", "mlp", "residual"]
    all_results = {}

    act_root = Path(args.activations_dir)
    # Discover models and disagreement types from directory structure
    for model_dir in sorted(act_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for dt_dir in sorted(model_dir.iterdir()):
            if not dt_dir.is_dir():
                continue
            dt_name = dt_dir.name

            labels = torch.load(dt_dir / "labels.pt", map_location="cpu", weights_only=True)
            print(f"\n{'='*60}")
            print(f"Model: {model_name}  Disagreement: {dt_name}  Samples: {len(labels)}")
            print(f"{'='*60}")

            for act_type in act_types:
                out_dir = Path(args.output_dir) / model_name / dt_name
                acc_path = out_dir / f"{act_type}_accuracies.json"

                if acc_path.exists():
                    print(f"  Skipping {act_type} (result already exists at {acc_path})")
                    with open(acc_path) as f:
                        all_results[f"{model_name}/{dt_name}/{act_type}"] = json.load(f)
                    continue

                act_path = dt_dir / f"{act_type}_activations.pt"
                if not act_path.exists():
                    print(f"  Skipping {act_type} (activation file not found)")
                    continue

                activations = torch.load(act_path, map_location="cpu", weights_only=True)
                print(f"  {act_type}: {activations.shape}")

                accuracies, probes = train_probes_for_activation(
                    activations, labels, act_type,
                    args.lr, args.epochs, args.batch_size, args.device, args.seed
                )

                # Store results
                key = f"{model_name}/{dt_name}/{act_type}"
                all_results[key] = accuracies

                # Save per-activation accuracy file and probe weights
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(acc_path, "w") as f:
                    json.dump(accuracies, f, indent=2)
                weights_path = out_dir / f"{act_type}_probes.pt"
                torch.save(probes, weights_path)
                print(f"  Saved {acc_path}")
                print(f"  Saved {weights_path}")

    # Train probes on combined (all disagreement types) per model
    for model_dir in sorted(act_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        dt_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

        if len(dt_dirs) < 2:
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name}  Disagreement: combined ({len(dt_dirs)} types)")
        print(f"{'='*60}")

        for act_type in act_types:
            out_dir = Path(args.output_dir) / model_name / "combined"
            acc_path = out_dir / f"{act_type}_accuracies.json"

            if acc_path.exists():
                print(f"  Skipping {act_type} (result already exists at {acc_path})")
                with open(acc_path) as f:
                    all_results[f"{model_name}/combined/{act_type}"] = json.load(f)
                continue

            # Concatenate activations and labels from all disagreement types
            act_parts = []
            label_parts = []
            for dt_dir in dt_dirs:
                act_path = dt_dir / f"{act_type}_activations.pt"
                labels_path = dt_dir / "labels.pt"
                if not act_path.exists() or not labels_path.exists():
                    continue
                act_parts.append(torch.load(act_path, map_location="cpu", weights_only=True))
                label_parts.append(torch.load(labels_path, map_location="cpu", weights_only=True))

            if not act_parts:
                print(f"  Skipping {act_type} (no activation files found)")
                continue

            activations = torch.cat(act_parts, dim=0)
            labels = torch.cat(label_parts, dim=0)
            print(f"  {act_type}: {activations.shape} (combined from {len(act_parts)} types)")

            accuracies, probes = train_probes_for_activation(
                activations, labels, act_type,
                args.lr, args.epochs, args.batch_size, args.device, args.seed
            )

            key = f"{model_name}/combined/{act_type}"
            all_results[key] = accuracies

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(acc_path, "w") as f:
                json.dump(accuracies, f, indent=2)
            weights_path = out_dir / f"{act_type}_probes.pt"
            torch.save(probes, weights_path)
            print(f"  Saved {acc_path}")
            print(f"  Saved {weights_path}")

    # Save summary: best accuracy per (model, disagreement_type, activation_type)
    summary = {}
    for key, accs in all_results.items():
        best_component = max(accs, key=accs.get)
        summary[key] = {
            "best_component": best_component,
            "best_accuracy": accs[best_component],
            "mean_accuracy": sum(accs.values()) / len(accs),
        }

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary saved to {summary_path}")
    print(f"{'='*60}")
    for key, info in sorted(summary.items()):
        print(f"  {key}: best={info['best_accuracy']:.2f}% ({info['best_component']})  mean={info['mean_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
