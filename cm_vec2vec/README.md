# CMVec2Vec: Conceptual Model Vector-to-Vector Translation

A comprehensive library for translating between different embedding spaces using the vec2vec approach. CMVec2Vec extends the original vec2vec framework to support multiple embedding domains including Natural Language, Conceptual Models, and other custom embedding spaces.

## Features

- 🚀 **Multi-Domain Support**: Translate between any embedding domains (NL, CM, custom)
- 🎯 **Unsupervised Learning**: No paired training data required
- 📊 **Comprehensive Evaluation**: All metrics from the vec2vec paper
- 🔧 **Modular Design**: Easy to extend and customize
- ⚡ **GPU Acceleration**: Optimized for CUDA training
- 📈 **Training Monitoring**: Built-in progress tracking and checkpointing
- 🎨 **Visualization**: Embedding visualization and analysis tools

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU training)

### Install Dependencies

```bash
pip install torch scikit-learn matplotlib tqdm numpy pandas toml
```

### Install CMVec2Vec

```bash
# Clone the repository
git clone <repository-url>
cd cm-vec2vec

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Basic Training

```python
from cm_vec2vec import CMVec2VecTranslator, CMVec2VecTrainer, load_cm_vec2vec_data

# Load data
train_loader, val_loader, test_loader = load_cm_vec2vec_data(
    data_path='data/embeddings.pkl',
    domains=['nl', 'cm']
)

# Create model
model = CMVec2VecTranslator(
    embedding_dims={'nl': 1536, 'cm': 1536},
    latent_dim=256,
    hidden_dim=512
)

# Create trainer
trainer = CMVec2VecTrainer(
    model=model,
    lr_generator=1e-4,
    lr_discriminator=4e-4
)

# Train
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_dir='checkpoints'
)
```

### 2. NL2CM Translation

```python
from cm_vec2vec import CMVec2VecTranslator, CMVec2VecEvaluator
from cm_vec2vec.data_loader import load_nl2cm_data

# Load NL2CM data
train_loader, val_loader, test_loader = load_nl2cm_data(
    'datasets/eamodelset_nl2cm_embeddings_df.pkl'
)

# Create and train model
model = CMVec2VecTranslator(
    embedding_dims={'NL_Serialization_Emb': 1536, 'CM_Serialization_Emb': 1536},
    latent_dim=256
)

trainer = CMVec2VecTrainer(model)
history = trainer.train(train_loader, val_loader, epochs=100)

# Evaluate
evaluator = CMVec2VecEvaluator(model)
results = evaluator.evaluate_all(nl_embeddings, cm_embeddings, 'NL_Serialization_Emb', 'CM_Serialization_Emb')
print(evaluator.create_evaluation_table(results))
```

### 3. Translation

```python
# Translate between domains
nl_embeddings = torch.randn(100, 1536)  # NL embeddings
cm_embeddings = model.translate(nl_embeddings, 'nl', 'cm')

# Batch translation
results = model.translate_batch(
    source_embeddings=nl_embeddings,
    source_domain='nl',
    target_domains=['cm', 'other_domain']
)
```

## Architecture

CMVec2Vec uses a modular architecture with four main components:

### 1. Adapters
Transform embeddings between original space and shared latent space:
- **Input Adapters**: `domain_embeddings → latent_representations`
- **Output Adapters**: `latent_representations → domain_embeddings`

### 2. Shared Backbone
Processes embeddings in the shared latent space:
- **Backbone**: `latent_representations → refined_latent_representations`

### 3. Discriminators
Adversarial networks for output and latent spaces:
- **Output Discriminators**: Distinguish real vs fake embeddings for each domain
- **Latent Discriminator**: Distinguish latent representations from different domains

### 4. Main Translator
Orchestrates all components and provides translation interface.

## Training Approach

The training process implements multi-objective optimization with five key loss functions:

### 1. Adversarial Loss
Ensures translated embeddings are indistinguishable from real target embeddings.

### 2. Reconstruction Loss
Maintains identity within each embedding space.

### 3. Cycle Consistency Loss
Enables round-trip translation fidelity.

### 4. Vector Space Preservation (VSP) Loss
Maintains pairwise relationships during translation.

### 5. Latent Adversarial Loss
Ensures latent representations are domain-agnostic.

## Configuration

CMVec2Vec uses TOML configuration files for easy customization:

```toml
[model]
embedding_dims = { nl = 1536, cm = 1536 }
latent_dim = 256
hidden_dim = 512
adapter_depth = 3
backbone_depth = 4

[training]
lr_generator = 1e-4
lr_discriminator = 4e-4
loss_weights = { reconstruction = 15.0, cycle_consistency = 15.0, vsp = 2.0, adversarial = 1.0 }
epochs = 100
batch_size = 32

[data]
data_path = "data/embeddings.pkl"
domains = ["nl", "cm"]
test_size = 0.2
val_size = 0.1
```

## Examples

### Example 1: Basic Training

```bash
python cm_vec2vec/examples/basic_training.py \
    --data_path data/embeddings.pkl \
    --domains nl cm \
    --epochs 100 \
    --batch_size 32
```

### Example 2: NL2CM Training

```bash
python cm_vec2vec/examples/nl2cm_training.py \
    --data_path datasets/eamodelset_nl2cm_embeddings_df.pkl \
    --epochs 100 \
    --config configs/nl2cm.toml
```

### Example 3: Evaluation

```bash
python cm_vec2vec/examples/evaluation_example.py \
    --model_path checkpoints/best_model.pt \
    --data_path data/embeddings.pkl \
    --domains nl cm \
    --save_plots
```

## API Reference

### Core Classes

#### `CMVec2VecTranslator`
Main translation model class.

```python
class CMVec2VecTranslator(nn.Module):
    def __init__(self, embedding_dims, latent_dim=256, hidden_dim=512, ...):
        """Initialize the translator."""
    
    def translate(self, embeddings, source_domain, target_domain, condition=None):
        """Translate embeddings from source to target domain."""
    
    def translate_batch(self, source_embeddings, source_domain, target_domains, condition=None):
        """Translate embeddings to multiple target domains."""
```

#### `CMVec2VecTrainer`
Training orchestrator with all loss functions.

```python
class CMVec2VecTrainer:
    def __init__(self, model, device='cuda', lr_generator=1e-4, ...):
        """Initialize the trainer."""
    
    def train(self, train_loader, val_loader, epochs, save_dir='checkpoints'):
        """Train the model."""
```

#### `CMVec2VecEvaluator`
Comprehensive evaluation with all vec2vec metrics.

```python
class CMVec2VecEvaluator:
    def __init__(self, model, device='cuda'):
        """Initialize the evaluator."""
    
    def evaluate_all(self, source_embeddings, target_embeddings, source_domain, target_domain):
        """Compute all evaluation metrics."""
```

## Evaluation Metrics

The library implements comprehensive evaluation metrics:

### Basic Translation Metrics
- **Cosine Similarity**: Mean cosine similarity between translated and target embeddings
- **Mean Rank**: Average rank of correct answers in retrieval
- **Top-K Accuracy**: Fraction of queries where correct answer is in top-K
- **MRR**: Mean Reciprocal Rank

### Advanced Metrics
- **Cycle Consistency**: Round-trip translation fidelity
- **Geometry Preservation**: Correlation of pairwise similarities
- **Clustering Performance**: ARI and NMI scores

## Advanced Features

### 1. Conditioning Support
Enable conditioning on additional information:

```python
model = CMVec2VecTranslator(
    embedding_dims={'nl': 1536, 'cm': 1536},
    use_conditioning=True,
    cond_dim=32
)

# Translation with condition
condition = torch.randn(batch_size, 32)
cm_embeddings = model.translate(nl_embeddings, 'nl', 'cm', condition)
```

### 2. Multi-Domain Support
Add new domains dynamically:

```python
model.add_domain('new_domain', embedding_dim=512)
```

### 3. Custom Loss Functions
Extend the trainer with custom loss functions:

```python
class CustomCMVec2VecTrainer(CMVec2VecTrainer):
    def compute_custom_loss(self, outputs, batch):
        # Implement your custom loss here
        return custom_loss
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Training Instability**
   - Reduce learning rates
   - Adjust loss weights
   - Use different GAN types

3. **Poor Translation Quality**
   - Increase model capacity
   - Increase training time
   - Adjust loss weights

## Citation

If you use this library, please cite the original vec2vec paper:

```bibtex
@article{jha2025harnessing,
  title={Harnessing the Universal Geometry of Embeddings},
  author={Jha, Akshita and Zhang, Yuchen and Shmatikov, Vitaly and Morris, Christopher},
  journal={arXiv preprint arXiv:2505.12540},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the vec2vec approach from Jha et al. (2025)
- Extends the original vec2vec framework for multi-domain translation
- Implements comprehensive evaluation and visualization tools
