# Sycophancy Detection in Large Language Models

This project provides tools for training linear probes to detect sycophancy and truthfulness in large language models by analyzing internal activations across different model components.

## Overview

The project trains linear classifiers (probes) on activations extracted from different components of LLMs:
- **Multi-Head Attention (MHA)** outputs
- **MLP layer** outputs  
- **Residual stream** activations

These probes are trained on the TruthfulQA dataset to identify patterns associated with sycophantic behavior versus truthful responses.

## Features

- **Activation Extraction**: Extract activations from MHA heads, MLP layers, and residual streams
- **Linear Probe Training**: Train binary classifiers on extracted activations
- **Visualization**: Generate heatmaps (for MHA) and line plots (for MLP/residual) showing accuracy across layers/heads
- **Batch Processing**: Support for distributed training via batch job submission
- **Multiple Models**: Compatible with Gemma and Llama model families

## Project Structure

```
.
├── linear_probe/
│   ├── train.py                    # Main training script for linear probes
│   ├── extract_activation.py       # Activation extraction utilities
│   ├── linear_probe.py             # Linear probe model definition
│   ├── linear_probe_data_utils.py  # Data preparation utilities
│   └── compute_proj_std.py         # Projection statistics computation
├── inference_mha.py                # Inference using MHA probes
├── inference_mlp.py                # Inference using MLP probes
├── inference_residual.py           # Inference using residual probes
├── submit_batch_job.py             # Batch job submission script
├── utils.py                        # General utility functions
├── attention_map.ipynb             # Attention visualization notebook
├── compute_metrics.ipynb           # Metrics computation notebook
├── get_prediction.ipynb            # Prediction analysis notebook
├── plots.ipynb                     # Results visualization notebook
└── pixi.toml                       # Pixi environment configuration
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Pixi package manager (or conda/pip)

### Installation

1. Clone the repository:
```bash
cd /path/to/sycophancy
```

2. Install dependencies using Pixi:
```bash
pixi install
```

Or manually install required packages:
```bash
pip install torch transformers datasets scikit-learn matplotlib pandas numpy tqdm
```

## Usage

### Training Linear Probes

Train probes on a specific model and activation type:

```bash
python train.py --model_id gemma-3 --activation_type mha
```

### Running Inference

After training, use the inference scripts to apply probes:

```bash
# For MHA probes
python inference_mha.py

# For MLP probes
python inference_mlp.py

# For residual stream probes
python inference_residual.py
```

### Batch Job Submission

For distributed training environments:

```bash
python submit_batch_job.py
```

## Output

The training process generates:

1. **Trained Probes**: Saved as `.pth` files in `{output_dir}/{model_name}/`
   - MHA: `linear_probe_{layer}_{head}.pth`
   - MLP: `linear_probe_mlp_{layer}.pth`
   - Residual: `linear_probe_residual_{layer}.pth`

2. **Accuracy Dictionary**: `accuracies_dict_{activation_type}.pkl` containing validation accuracies for each component

3. **Visualizations**:
   - MHA: Heatmap showing accuracy across all layers and heads
   - MLP/Residual: Line plot showing accuracy across layers

## Dataset

The project uses the [TruthfulQA dataset](https://huggingface.co/datasets/truthfulqa/truthful_qa) for training and evaluation:
- Training/validation split: 80/20

## Model Support

Currently tested with:
- Google Gemma models (gemma-2, gemma-3)
- Meta Llama models (Llama-3.2)

The code is designed to be extensible to other transformer-based language models.

## Notebooks

Interactive Jupyter notebooks are provided for:
- `attention_map.ipynb`: Visualizing attention patterns
- `compute_metrics.ipynb`: Computing evaluation metrics
- `get_prediction.ipynb`: Analyzing model predictions
- `plots.ipynb`: Creating custom visualizations

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## Contact

[Add contact information here]
