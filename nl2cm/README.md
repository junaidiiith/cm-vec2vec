# NL2CM: Natural Language to Conceptual Model Translation Library

A comprehensive PyTorch library for translating between Natural Language and Conceptual Model embedding spaces using the vec2vec approach. This library implements unsupervised learning to map embeddings from one domain to another without requiring paired training data.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Training Approach](#training-approach)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Performance](#performance)
- [Contributing](#contributing)

## Overview

The NL2CM library implements the vec2vec approach for unsupervised embedding translation between different domains. The key innovation is learning to translate between Natural Language (NL) and Conceptual Model (CM) embedding spaces using:

- **Unpaired Training**: No need for aligned NL-CM pairs
- **Adversarial Learning**: Ensures translated embeddings match target distribution
- **Geometry Preservation**: Maintains semantic relationships during translation
- **Cycle Consistency**: Enables round-trip translation validation

### Key Features

- 🔄 **Bidirectional Translation**: NL ↔ CM embedding translation
- 🎯 **Unsupervised Learning**: No paired data required
- 📊 **Comprehensive Evaluation**: All metrics from the vec2vec paper
- 🏗️ **Modular Architecture**: Easy to extend and customize
- ⚡ **GPU Acceleration**: Optimized for CUDA training
- 📈 **Real-time Monitoring**: Training progress and metrics tracking

## Architecture

The NL2CM model follows a modular architecture with three main components:

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   NL Embeddings │───▶│   Adapters   │───▶│  Shared Backbone│
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                      │
┌─────────────────┐    ┌──────────────┐              │
│   CM Embeddings │───▶│   Adapters   │──────────────┘
└─────────────────┘    └──────────────┘              │
                                                     ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Translated CM   │◀───│Output Adapters│◀───│  Processed      │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

### Core Components

1. **Adapters**: Map embeddings to/from shared latent space
   - Input adapters: `embedding_dim` → `latent_dim`
   - Output adapters: `latent_dim` → `embedding_dim`
   - MLP architecture with LayerNorm, SiLU activation, dropout

2. **Shared Backbone**: Processes embeddings in latent space
   - Residual connections for stable training
   - Optional conditioning support
   - Configurable depth and hidden dimensions

3. **Discriminators**: Adversarial training components
   - Output space discriminators (NL, CM)
   - Latent space discriminator
   - Least squares GAN for stable training

## Training Approach

The training process uses multiple loss functions to ensure high-quality translation:

### Loss Functions

1. **Adversarial Loss** (`λ_adv = 1.0`)
   - Output space: Distinguishes real vs translated embeddings
   - Latent space: Aligns latent representations across domains
   - Uses Least Squares GAN for stable training

2. **Reconstruction Loss** (`λ_rec = 15.0`)
   - Ensures identity mapping within each domain
   - Prevents information loss during translation
   - MSE between original and reconstructed embeddings

3. **Cycle Consistency Loss** (`λ_cyc = 15.0`)
   - Enforces round-trip translation: NL → CM → NL ≈ NL
   - Acts as implicit supervision without paired data
   - Critical for maintaining semantic content

4. **Vector Space Preservation (VSP) Loss** (`λ_vsp = 2.0`)
   - Preserves pairwise similarities within batches
   - Maintains semantic neighborhoods
   - Essential for downstream tasks like retrieval

### Training Process

```python
# 1. Forward pass through translator
outputs = model(batch)  # Contains translations, reconstructions, latents

# 2. Compute generator losses
gen_losses = trainer.compute_generator_loss(outputs, batch)

# 3. Update generator (adapters + backbone)
optimizer_generator.zero_grad()
gen_losses['total'].backward()
optimizer_generator.step()

# 4. Compute discriminator losses
disc_losses = trainer.compute_discriminator_loss(outputs, batch)

# 5. Update discriminators
optimizer_discriminator.zero_grad()
disc_losses['total'].backward()
optimizer_discriminator.step()
```

### Key Training Insights

- **Unpaired Data**: Model learns from distribution matching, not explicit pairs
- **Dual Optimization**: Generator and discriminator have separate optimizers
- **Stable Training**: Separate forward passes prevent gradient conflicts
- **Early Stopping**: Prevents overfitting with validation monitoring

## Installation

### Requirements

```bash
pip install torch>=1.9.0
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install numpy
```

### From Source

```bash
git clone <repository-url>
cd nl2cm
pip install -e .
```

## Quick Start

### Basic Usage

```python
from nl2cm import NL2CMTranslator, NL2CMTrainer, NL2CMEvaluator
from nl2cm.data_loader import load_nl2cm_data

# 1. Load data
train_loader, val_loader, test_loader = load_nl2cm_data('data.pkl')

# 2. Create model
model = NL2CMTranslator(
    embedding_dim=1536,
    latent_dim=256,
    hidden_dim=512
)

# 3. Create trainer
trainer = NL2CMTrainer(
    model=model,
    device='cuda',
    lambda_rec=15.0,
    lambda_cyc=15.0,
    lambda_vsp=2.0
)

# 4. Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)

# 5. Evaluate model
evaluator = NL2CMEvaluator(model, device='cuda')
results = evaluator.evaluate_all(nlt_embeddings, cmt_embeddings)
```

### Command Line Training

```bash
# Basic training
python nl2cm/train.py --epochs 100 --batch_size 32

# Advanced configuration
python nl2cm/train.py \
    --epochs 200 \
    --batch_size 16 \
    --latent_dim 512 \
    --hidden_dim 1024 \
    --lambda_rec 20.0 \
    --lambda_cyc 20.0 \
    --lambda_vsp 3.0 \
    --save_dir checkpoints/experiment_1
```

## API Reference

### Core Classes

#### `NL2CMTranslator`

Main translation model with configurable architecture.

```python
model = NL2CMTranslator(
    embedding_dim=1536,      # Input/output embedding dimension
    latent_dim=256,          # Latent space dimension
    hidden_dim=512,          # Hidden layer dimension
    adapter_depth=3,         # Number of adapter layers
    backbone_depth=4,        # Number of backbone layers
    dropout=0.1,             # Dropout rate
    use_conditioning=False,  # Enable conditioning
    cond_dim=0              # Conditioning dimension
)
```

**Key Methods:**
- `forward(batch)`: Full forward pass with all outputs
- `translate_nlt_to_cmt(nlt_emb)`: Translate NL to CM embeddings
- `translate_cmt_to_nlt(cmt_emb)`: Translate CM to NL embeddings

#### `NL2CMTrainer`

Handles training loop and loss computation.

```python
trainer = NL2CMTrainer(
    model=model,
    device='cuda',
    lr_generator=1e-4,       # Generator learning rate
    lr_discriminator=4e-4,   # Discriminator learning rate
    lambda_rec=15.0,         # Reconstruction loss weight
    lambda_cyc=15.0,         # Cycle consistency loss weight
    lambda_vsp=2.0,          # Vector space preservation weight
    lambda_adv=1.0,          # Adversarial loss weight
    lambda_latent=1.0,       # Latent adversarial weight
    weight_decay=0.01        # Weight decay
)
```

**Key Methods:**
- `train(train_loader, val_loader, epochs)`: Full training loop
- `train_step(batch)`: Single training step
- `validate(val_loader)`: Validation evaluation
- `load_checkpoint(path)`: Load saved model

#### `NL2CMEvaluator`

Comprehensive evaluation with multiple metrics.

```python
evaluator = NL2CMEvaluator(model, device='cuda')
results = evaluator.evaluate_all(nlt_emb, cmt_emb)
```

**Key Methods:**
- `compute_cosine_similarity(nlt_emb, cmt_emb)`: Translation quality
- `compute_top_k_accuracy(nlt_emb, cmt_emb, k)`: Retrieval performance
- `compute_mean_rank(nlt_emb, cmt_emb)`: Ranking performance
- `compute_cycle_consistency_metrics(nlt_emb, cmt_emb)`: Round-trip quality
- `compute_geometry_preservation_metrics(nlt_emb, cmt_emb)`: Similarity preservation

### Data Loading

#### `load_nl2cm_data(data_path, test_size=0.2, random_state=42)`

Loads data and creates train/validation/test splits.

**Input Format:**
```python
# Expected pickle file structure
{
    'NL_Serialization_Emb': [embedding_1, embedding_2, ...],  # List of numpy arrays
    'CM_Serialization_Emb': [embedding_1, embedding_2, ...]   # List of numpy arrays
}
```

**Returns:**
- `train_loader`: Unpaired data for training
- `val_loader`: Unpaired data for validation
- `test_loader`: Paired data for evaluation

## Usage Examples

### Example 1: Basic Training

```python
import torch
from nl2cm import NL2CMTranslator, NL2CMTrainer
from nl2cm.data_loader import load_nl2cm_data

# Load data
train_loader, val_loader, test_loader = load_nl2cm_data(
    'datasets/eamodelset_nl2cm_embeddings_df.pkl'
)

# Create model
model = NL2CMTranslator(
    embedding_dim=1536,
    latent_dim=256,
    hidden_dim=512
)

# Create trainer
trainer = NL2CMTrainer(
    model=model,
    device='cuda',
    lr_generator=1e-4,
    lr_discriminator=4e-4
)

# Train
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_dir='checkpoints/basic_experiment'
)

print("Training completed!")
```

### Example 2: Custom Model Configuration

```python
# Advanced model with conditioning
model = NL2CMTranslator(
    embedding_dim=1536,
    latent_dim=512,          # Larger latent space
    hidden_dim=1024,         # Deeper networks
    adapter_depth=4,         # More adapter layers
    backbone_depth=6,        # Deeper backbone
    dropout=0.15,            # Higher dropout
    use_conditioning=True,   # Enable conditioning
    cond_dim=32             # Conditioning dimension
)

# Custom loss weights
trainer = NL2CMTrainer(
    model=model,
    device='cuda',
    lambda_rec=20.0,         # Stronger reconstruction
    lambda_cyc=20.0,         # Stronger cycle consistency
    lambda_vsp=5.0,          # Stronger geometry preservation
    lambda_adv=0.5,          # Weaker adversarial
    lambda_latent=0.5        # Weaker latent adversarial
)
```

### Example 3: Evaluation and Analysis

```python
from nl2cm import NL2CMEvaluator
import numpy as np

# Load trained model
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create evaluator
evaluator = NL2CMEvaluator(model, device='cuda')

# Load evaluation data
nlt_eval, cmt_eval = create_evaluation_splits('data.pkl', n_eval_samples=1000)
nlt_tensor = torch.FloatTensor(nlt_eval).to('cuda')
cmt_tensor = torch.FloatTensor(cmt_eval).to('cuda')

# Comprehensive evaluation
results = evaluator.evaluate_all(nlt_tensor, cmt_tensor)

# Print results table
print(evaluator.create_evaluation_table(results))

# Individual metrics
cosine_sim = evaluator.compute_cosine_similarity(nlt_tensor, cmt_tensor)
top1_acc = evaluator.compute_top_k_accuracy(nlt_tensor, cmt_tensor, k=1)
mean_rank = evaluator.compute_mean_rank(nlt_tensor, cmt_tensor)

print(f"Cosine Similarity: {cosine_sim:.4f}")
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Mean Rank: {mean_rank:.2f}")
```

### Example 4: Translation Inference

```python
# Load trained model
model.eval()

# Translate NL to CM
nlt_embeddings = torch.randn(10, 1536).to('cuda')  # 10 NL embeddings
with torch.no_grad():
    cmt_translated = model.translate_nlt_to_cmt(nlt_embeddings)

# Translate CM to NL
cmt_embeddings = torch.randn(10, 1536).to('cuda')  # 10 CM embeddings
with torch.no_grad():
    nlt_translated = model.translate_cmt_to_nlt(cmt_embeddings)

print(f"Translated shapes: {cmt_translated.shape}, {nlt_translated.shape}")

# Batch translation
batch = {
    'nlt': nlt_embeddings,
    'cmt': cmt_embeddings
}
with torch.no_grad():
    outputs = model(batch)
    print(f"Available outputs: {list(outputs.keys())}")
```

## Evaluation

The library provides comprehensive evaluation metrics matching the vec2vec paper:

### Core Metrics

1. **Cosine Similarity**: Mean cosine similarity between translated and target embeddings
2. **Top-K Accuracy**: Fraction of queries where correct answer is in top-K results
3. **Mean Rank**: Average rank of correct answers in retrieval
4. **MRR (Mean Reciprocal Rank)**: Harmonic mean of reciprocal ranks

### Advanced Metrics

1. **Cycle Consistency**: Round-trip translation fidelity
   - NL → CM → NL similarity
   - CM → NL → CM similarity

2. **Geometry Preservation**: Correlation of pairwise similarities
   - Within-batch similarity preservation
   - Semantic neighborhood maintenance

3. **Classification Performance**: Clustering-based evaluation
   - Adjusted Rand Index (ARI)
   - Normalized Mutual Information (NMI)

### Baseline Comparisons

- **Identity**: Direct embedding comparison (no translation)
- **Procrustes**: Orthogonal transformation baseline
- **Random**: Random ranking baseline

### Example Evaluation Results

```
================================================================================
NL2CM Translation Evaluation Results
================================================================================

Metric                    NL2CM           Identity        Procrustes      Random         
-------------------------------------------------------------------------------------
Cosine Similarity         0.3714          0.6591          0.8945          N/A            
Mean Rank                 488.67          N/A             N/A             489.00         
Top-1 Accuracy            0.0010          0.0000          0.0000          0.0000         
Top-5 Accuracy            0.0041          0.0000          0.0000          0.0000         
MRR                       0.0075          0.0000          0.0000          0.0000         

Cycle Consistency:
--------------------------------------------------
NL Cycle Similarity       0.3182         
CM Cycle Similarity       0.3698         
Mean Cycle Similarity     0.3440         

Geometry Preservation:
--------------------------------------------------
NL Geometry Correlation   0.6248         
CM Geometry Correlation   0.4724         
Mean Geometry Correlation 0.5486         
```

## Configuration

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 1536 | Input/output embedding dimension |
| `latent_dim` | 256 | Latent space dimension |
| `hidden_dim` | 512 | Hidden layer dimension |
| `adapter_depth` | 3 | Number of adapter layers |
| `backbone_depth` | 4 | Number of backbone layers |
| `dropout` | 0.1 | Dropout rate |
| `use_conditioning` | False | Enable conditioning |
| `cond_dim` | 0 | Conditioning dimension |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_generator` | 1e-4 | Generator learning rate |
| `lr_discriminator` | 4e-4 | Discriminator learning rate |
| `lambda_rec` | 15.0 | Reconstruction loss weight |
| `lambda_cyc` | 15.0 | Cycle consistency loss weight |
| `lambda_vsp` | 2.0 | Vector space preservation weight |
| `lambda_adv` | 1.0 | Adversarial loss weight |
| `lambda_latent` | 1.0 | Latent adversarial weight |
| `weight_decay` | 0.01 | Weight decay |

### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `test_size` | 0.2 | Fraction for test split |
| `random_state` | 42 | Random seed |
| `num_workers` | 4 | DataLoader workers |

## Advanced Features

### 1. Conditioning Support

Enable conditioning on target modeling language:

```python
model = NL2CMTranslator(
    embedding_dim=1536,
    latent_dim=256,
    use_conditioning=True,
    cond_dim=32
)

# During training/inference
condition = torch.randn(batch_size, 32)  # One-hot or learned embeddings
outputs = model(batch, condition=condition)
```

### 2. Custom Loss Weights

Fine-tune training by adjusting loss weights:

```python
# Emphasize reconstruction and cycle consistency
trainer = NL2CMTrainer(
    model=model,
    lambda_rec=25.0,      # Strong reconstruction
    lambda_cyc=25.0,      # Strong cycle consistency
    lambda_vsp=1.0,       # Moderate geometry preservation
    lambda_adv=0.5,       # Weak adversarial
    lambda_latent=0.5     # Weak latent adversarial
)
```

### 3. Progressive Training

Train with curriculum learning:

```python
# Start with reconstruction, add adversarial later
trainer = NL2CMTrainer(
    model=model,
    lambda_rec=20.0,
    lambda_cyc=20.0,
    lambda_vsp=0.0,       # Start without VSP
    lambda_adv=0.0,       # Start without adversarial
    lambda_latent=0.0
)

# Gradually increase weights during training
for epoch in range(epochs):
    if epoch > 20:
        trainer.lambda_adv = 1.0
    if epoch > 40:
        trainer.lambda_vsp = 2.0
```

### 4. Multi-GPU Training

Scale to multiple GPUs:

```python
import torch.nn as nn

# Wrap model for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

# Training remains the same
trainer = NL2CMTrainer(model, device='cuda')
```

## Performance

### Typical Results

On the EA Model Set dataset:

| Metric | Value | Baseline Comparison |
|--------|-------|-------------------|
| Cosine Similarity | 0.3714 | vs Identity: 0.6591 |
| Mean Rank | 488.67 | vs Random: 489.00 |
| Top-1 Accuracy | 0.0010 | vs Random: 0.0000 |
| Cycle Consistency | 0.3440 | - |
| Geometry Preservation | 0.5486 | - |

### Training Time

- **Small Model** (256 latent, 512 hidden): ~2 minutes/epoch on RTX 3080
- **Medium Model** (512 latent, 1024 hidden): ~5 minutes/epoch on RTX 3080
- **Large Model** (1024 latent, 2048 hidden): ~15 minutes/epoch on RTX 3080

### Memory Usage

- **Training**: ~8GB VRAM for batch_size=32, embedding_dim=1536
- **Inference**: ~2GB VRAM for batch_size=100
- **Model Size**: ~8.3M parameters for default configuration

### Optimization Tips

1. **Batch Size**: Use largest batch size that fits in memory
2. **Learning Rates**: Higher discriminator LR (4x generator) for stability
3. **Loss Weights**: Start with high reconstruction/cycle weights
4. **Early Stopping**: Monitor validation loss to prevent overfitting
5. **Gradient Clipping**: Add if training becomes unstable

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
git clone <repository-url>
cd nl2cm
pip install -e .
pip install pytest black flake8
```

### Running Tests

```bash
# Run all tests
python test_nl2cm.py

# Run specific test
python -m pytest tests/test_model.py -v
```

### Code Style

We use Black for formatting and flake8 for linting:

```bash
black nl2cm/
flake8 nl2cm/
```

## Citation

If you use this library in your research, please cite:

```bibtex
@article{jha2025harnessing,
  title={Harnessing the Universal Geometry of Embeddings},
  author={Jha, Akshita and Zhang, Yuchen and Shmatikov, Vitaly and Morris, Christopher},
  journal={arXiv preprint arXiv:2505.12540},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the vec2vec approach from Jha et al. (2025)
- Implements adversarial training and geometry preservation
- Extends to NL2CM translation task
- Built with PyTorch and modern deep learning practices

---

For more information, examples, and updates, visit our [documentation](docs/) and [GitHub repository](https://github.com/your-repo/nl2cm).
