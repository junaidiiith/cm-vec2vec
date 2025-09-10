# TO-v2vGAN: Task-Operator vec2vec GAN

This repository implements the **Task-Operator vec2vec GAN (TO-v2vGAN)** as described in the paper: "Translating Embeddings of Natural-Language Domain Descriptions into Embeddings of Corresponding Domain Models."

## Overview

TO-v2vGAN learns to translate embeddings of natural-language domain descriptions into embeddings of corresponding domain models (e.g., UML/ArchiMate/OntoUML graphs). The key innovation is the decomposition into:

1. **Shared latent space** where description/model semantics align
2. **Explicit task-operator** that captures the modeling transformation

This approach enables:
- **Data-driven domain modeling** through retrieval and classification
- **Zero-shot generalization** across domains
- **Geometry preservation** in vector spaces
- **Exploitation of unpaired data** through adversarial training

## Architecture

The model consists of:

- **Adapters**: Transform domain descriptions (X) and models (Y) to shared latent space
- **Task-Operator**: Decomposed as `T(z) = P*z + Oper(z)` where:
  - `P` is near-orthogonal (shared semantics)
  - `Oper` is low-rank residual (task transformation)
- **Generators**: Fx (X→Y), Fy (Y→X), Rx (X→X), Ry (Y→Y)
- **Discriminators**: Output space (Dx, Dy) and latent space (D_latent_1, D_latent_2)

## Loss Functions

The complete objective combines:

- **Adversarial**: Output + latent space GAN losses
- **Reconstruction**: Autoencoder reconstruction
- **Cycle Consistency**: Fx(Fy(y)) ≈ y, Fy(Fx(x)) ≈ x
- **Vector Space Preservation**: Maintain pairwise dot products
- **Operator Priors**: Orthogonality + low-rank constraints
- **Paired Supervision**: Optional InfoNCE/Huber loss

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cm-vec2vec
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Synthetic Data

Train on synthetic embeddings to test the implementation:

```bash
python train.py --x_dim 768 --y_dim 256 --epochs 100 --batch_size 32
```

### Training with Configuration File

Use the provided configuration:

```bash
python train.py --config config.yaml
```

### Custom Training

Modify `config.yaml` or use command-line arguments:

```bash
python train.py \
    --x_dim 768 \
    --y_dim 256 \
    --latent_dim 128 \
    --hidden_dim 256 \
    --epochs 200 \
    --lr 1e-4 \
    --lambda_rec 15.0 \
    --lambda_cyc 15.0
```

## Data Format

### Input Embeddings

- **Domain Descriptions (X)**: Natural language embeddings (e.g., BERT, GTE, E5)
- **Domain Models (Y)**: Graph/model embeddings (e.g., GIN, GraphSAGE, WL-kernel)

### Data Structure

```python
# X: Domain description embeddings
x_embeddings = torch.randn(batch_size, x_dim)  # e.g., (32, 768)

# Y: Domain model embeddings  
y_embeddings = torch.randn(batch_size, y_dim)  # e.g., (32, 256)

# Optional: Paired supervision
paired_data = [(x_i, y_i), ...]  # Small set of aligned pairs
```

## Training

### Basic Training Loop

```python
from src.models import TOv2vGAN
from src.training import TOv2vGANTrainer

# Create model
model = TOv2vGAN(x_dim=768, y_dim=256, latent_dim=128)

# Create trainer
trainer = TOv2vGANTrainer(model, x_dim=768, y_dim=256)

# Train
losses = trainer.train(x_loader, y_loader, epochs=100)
```

### Advanced Training

```python
# Custom loss weights
trainer = TOv2vGANTrainer(
    model, x_dim=768, y_dim=256,
    lambda_rec=15.0,      # Stronger reconstruction
    lambda_cyc=15.0,      # Stronger cycle consistency
    lambda_vsp=2.0,       # Stronger geometry preservation
    lambda_op=2.0,        # Stronger operator priors
    lambda_pair=5.0       # Stronger paired supervision
)

# Multiple discriminator steps
losses = trainer.train(x_loader, y_loader, epochs=100, n_d_steps=2)
```

## Evaluation

### Comprehensive Metrics

The evaluator computes all metrics from the paper:

```python
from src.evaluation import TOv2vGANEvaluator

evaluator = TOv2vGANEvaluator(model, device='cuda')

# Evaluate all metrics
results = evaluator.evaluate_all(x_val, y_val, y_labels)

# Print results
evaluator.print_results(results)

# Save report
from src.evaluation import create_evaluation_report
report = create_evaluation_report(results, 'evaluation_report.txt')
```

### Metrics Included

- **Retrieval**: R@1, R@5, R@10, MRR
- **Classification**: Accuracy, Macro-F1
- **Geometry**: VSP correlation, cycle cosine similarity
- **Distribution**: Maximum Mean Discrepancy (MMD)

## Model Analysis

### Operator Factorization

```python
# Access task-operator components
model = TOv2vGAN(...)

# Get operator losses
op_losses = model.get_operator_losses()
orthogonality_loss = op_losses['orthogonality']
rank_loss = op_losses['rank']

# Analyze P matrix (shared semantics)
P = model.T.P
singular_values = torch.svd(P)[1]  # Should be near 1.0

# Analyze Oper (task transformation)
U, V = model.T.U, model.T.V
rank = min(U.shape[1], V.shape[1])  # Should be low
```

### Latent Space Analysis

```python
# Extract latent representations
with torch.no_grad():
    z_x = model.Ax(x_embeddings)           # X → latent
    z_y = model.Ay(y_embeddings)           # Y → latent
    z_x_transformed = model.T(z_x)         # After task-operator
    z_y_transformed = model.T(z_y)         # After task-operator

# Analyze alignment
latent_similarity = F.cosine_similarity(z_x_transformed, z_y_transformed)
```

## Extending to Other Tasks

TO-v2vGAN generalizes to various text→structure tasks:

```python
# Requirements → Architecture
model = TOv2vGAN(x_dim=768, y_dim=512)  # Text → ArchiMate

# Legal facts → Rule graphs  
model = TOv2vGAN(x_dim=768, y_dim=256)  # Text → Legal KG

# Bug reports → Patch embeddings
model = TOv2vGAN(x_dim=768, y_dim=128)  # Text → Code vectors

# Dataset cards → Schema/ER
model = TOv2vGAN(x_dim=768, y_dim=384)  # Text → Database schema
```

## Hyperparameter Tuning

### Key Parameters

- **Latent Dimension**: 64-256 (balance expressiveness vs. efficiency)
- **Loss Weights**: Start with paper defaults, adjust based on task
- **Learning Rate**: 1e-4 to 5e-4 (use paper's 2e-4 as baseline)
- **Batch Size**: 16-64 (larger for stability, smaller for memory)

### Recommended Settings

```yaml
# For high-quality embeddings (BERT, GTE)
lambda_rec: 15.0
lambda_cyc: 15.0
lambda_vsp: 2.0

# For graph embeddings (GIN, GraphSAGE)  
lambda_rec: 10.0
lambda_cyc: 10.0
lambda_vsp: 1.0

# For low-pair scenarios
lambda_pair: 5.0
lambda_op: 2.0
```

## Troubleshooting

### Common Issues

1. **Mode Collapse**: Increase `lambda_rec` and `lambda_cyc`
2. **Poor Geometry**: Increase `lambda_vsp` and `lambda_op`
3. **Training Instability**: Use gradient clipping, adjust learning rate
4. **Memory Issues**: Reduce batch size, use gradient accumulation

### Debugging

```python
# Monitor individual losses
losses = trainer.train_step(x, y)
print(f"Generator: {losses['total_g']:.4f}")
print(f"Discriminator: {losses['d_adv']:.4f}")
print(f"Reconstruction: {losses['rec']:.4f}")
print(f"Cycle: {losses['cyc']:.4f}")
print(f"VSP: {losses['vsp']:.4f}")
print(f"Operator: {losses['ortho'] + losses['rank']:.4f}")
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{tov2vgan2024,
  title={Task-Operator vec2vec GAN: Translating Embeddings of Natural-Language Domain Descriptions into Embeddings of Corresponding Domain Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation is based on the research paper "Task-Operator vec2vec GAN" and builds upon concepts from:
- CycleGAN and pix2pix for image-to-image translation
- WGAN-GP for stable adversarial training
- Contrastive learning for cross-modal alignment
- Graph neural networks for model embeddings
