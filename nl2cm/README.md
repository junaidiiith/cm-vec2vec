# NL2CM: Natural Language to Conceptual Model Translation

A comprehensive library for translating between Natural Language (NL) and Conceptual Model (CM) embedding spaces using the vec2vec approach. This library implements unsupervised translation between different embedding domains without requiring paired training data.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Training Approach](#training-approach)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Detailed Usage](#detailed-usage)
7. [API Reference](#api-reference)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Examples](#examples)
10. [Advanced Features](#advanced-features)
11. [Troubleshooting](#troubleshooting)

## Overview

The NL2CM library implements a sophisticated translation system that learns to map between Natural Language and Conceptual Model embedding spaces. Based on the vec2vec approach, it uses:

- **Unpaired Training**: No need for aligned NL-CM pairs
- **Adversarial Learning**: Ensures translated embeddings match target distributions
- **Geometry Preservation**: Maintains semantic relationships during translation
- **Cycle Consistency**: Enables round-trip translation fidelity

### Key Features

- 🚀 **Unsupervised Translation**: Learn from unpaired embedding data
- 🎯 **Multi-Modal Support**: Handle different embedding dimensions and types
- 📊 **Comprehensive Evaluation**: All metrics from the vec2vec paper
- 🔧 **Modular Design**: Easy to extend and customize
- ⚡ **GPU Acceleration**: Optimized for CUDA training
- 📈 **Training Monitoring**: Built-in progress tracking and checkpointing

## Architecture

The NL2CM system uses a modular architecture with four main components:

### 1. Adapters (`Adapter`)
Transform embeddings between the original space and a shared latent space:

```python
# Input Adapters: embedding_space → latent_space
nlt_adapter: NL_embeddings → latent_representations
cmt_adapter: CM_embeddings → latent_representations

# Output Adapters: latent_space → embedding_space  
nlt_output_adapter: latent_representations → NL_embeddings
cmt_output_adapter: latent_representations → CM_embeddings
```

**Architecture Details:**
- MLP with configurable depth (default: 3 layers)
- LayerNorm + SiLU activation
- Kaiming weight initialization
- Optional dropout for regularization

### 2. Shared Backbone (`SharedBackbone`)
Processes embeddings in the shared latent space:

```python
# Shared processing in latent space
backbone: latent_representations → refined_latent_representations
```

**Architecture Details:**
- Configurable depth (default: 4 layers)
- Residual connections for stable training
- Optional conditioning support
- LayerNorm + SiLU activation

### 3. Discriminators (`Discriminator`)
Adversarial networks for output and latent spaces:

```python
# Output space discriminators
nlt_discriminator: NL_embeddings → real/fake_score
cmt_discriminator: CM_embeddings → real/fake_score

# Latent space discriminator
latent_discriminator: latent_representations → real/fake_score
```

**Architecture Details:**
- MLP with LeakyReLU activation
- LayerNorm for stability
- Single output for binary classification

### 4. Main Translator (`NL2CMTranslator`)
Orchestrates all components:

```python
class NL2CMTranslator(nn.Module):
    def __init__(self, embedding_dim=1536, latent_dim=256, ...):
        # Initialize all components
        self.nlt_adapter = Adapter(...)
        self.cmt_adapter = Adapter(...)
        self.backbone = SharedBackbone(...)
        self.nlt_output_adapter = Adapter(...)
        self.cmt_output_adapter = Adapter(...)
        self.nlt_discriminator = Discriminator(...)
        self.cmt_discriminator = Discriminator(...)
        self.latent_discriminator = Discriminator(...)
```

## Training Approach

The training process implements a multi-objective optimization with five key loss functions:

### 1. Adversarial Loss
Ensures translated embeddings are indistinguishable from real target embeddings:

```python
# Output space adversarial loss
L_adv_output = L_GAN(D_NLT, F_C2N) + L_GAN(D_CMT, F_N2C)

# Latent space adversarial loss  
L_adv_latent = L_GAN(D_latent, T∘A_NLT) + L_GAN(D_latent, T∘A_CMT)

# Total adversarial loss
L_adv = λ_adv * L_adv_output + λ_latent * L_adv_latent
```

**Implementation:**
- Least Squares GAN for stable training
- Separate discriminators for each space
- Generator tries to fool discriminators

### 2. Reconstruction Loss
Maintains identity within each embedding space:

```python
L_rec = E[||R_NLT(x) - x||²] + E[||R_CMT(y) - y||²]
```

**Purpose:**
- Prevents information loss during translation
- Ensures adapters preserve embedding content
- Acts as regularization

### 3. Cycle Consistency Loss
Enables round-trip translation fidelity:

```python
L_cyc = E[||F_CMT→NLT(F_NLT→CMT(x)) - x||²] + E[||F_NLT→CMT(F_CMT→NLT(y)) - y||²]
```

**Purpose:**
- Implicit supervision without paired data
- Ensures translations are invertible
- Prevents mode collapse

### 4. Vector Space Preservation (VSP) Loss
Maintains pairwise relationships during translation:

```python
L_vsp = (1/B) * Σᵢⱼ ||xᵢ·xⱼ - F_NLT→CMT(xᵢ)·F_NLT→CMT(xⱼ)||² + 
        (1/B) * Σᵢⱼ ||yᵢ·yⱼ - F_CMT→NLT(yᵢ)·F_CMT→NLT(yⱼ)||²
```

**Purpose:**
- Preserves semantic neighborhoods
- Maintains embedding geometry
- Critical for downstream tasks

### 5. Training Procedure

The training alternates between generator and discriminator updates:

```python
for epoch in range(epochs):
    for batch in dataloader:
        # Generator update
        outputs = model(batch)
        gen_losses = compute_generator_loss(outputs, batch)
        optimizer_generator.zero_grad()
        gen_losses['total'].backward()
        optimizer_generator.step()
        
        # Discriminator update
        outputs_disc = model(batch)  # Fresh forward pass
        disc_losses = compute_discriminator_loss(outputs_disc, batch)
        optimizer_discriminator.zero_grad()
        disc_losses['total'].backward()
        optimizer_discriminator.step()
```

**Loss Weighting:**
- Reconstruction: λ_rec = 15.0 (high priority)
- Cycle Consistency: λ_cyc = 15.0 (high priority)
- Vector Space Preservation: λ_vsp = 2.0 (medium priority)
- Adversarial: λ_adv = 1.0 (low priority)
- Latent Adversarial: λ_latent = 1.0 (low priority)

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU training)

### Install Dependencies

```bash
pip install torch scikit-learn matplotlib tqdm numpy pandas
```

### Install NL2CM

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
from nl2cm import NL2CMTranslator, NL2CMTrainer, load_nl2cm_data

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

### 2. Evaluation

```python
from nl2cm import NL2CMEvaluator

# Create evaluator
evaluator = NL2CMEvaluator(model, device='cuda')

# Evaluate
results = evaluator.evaluate_all(nlt_embeddings, cmt_embeddings)

# Print results
print(evaluator.create_evaluation_table(results))
```

### 3. Translation

```python
# Translate NL to CM
cm_embeddings = model.translate_nlt_to_cmt(nl_embeddings)

# Translate CM to NL  
nl_embeddings = model.translate_cmt_to_nlt(cm_embeddings)
```

## Detailed Usage

### Data Loading

The library supports flexible data loading with automatic train/validation/test splits:

```python
from nl2cm import load_nl2cm_data, create_evaluation_splits

# Load with custom splits
train_loader, val_loader, test_loader = load_nl2cm_data(
    data_path='path/to/embeddings.pkl',
    test_size=0.2,
    random_state=42
)

# Create evaluation data
nlt_eval, cmt_eval = create_evaluation_splits(
    data_path='path/to/embeddings.pkl',
    n_eval_samples=1000
)
```

**Expected Data Format:**
```python
# Pickle file containing pandas DataFrame with columns:
df = {
    'NL_Serialization_Emb': [embedding_arrays],  # Shape: (N, embedding_dim)
    'CM_Serialization_Emb': [embedding_arrays],  # Shape: (N, embedding_dim)
    # ... other columns
}
```

### Model Configuration

```python
model = NL2CMTranslator(
    embedding_dim=1536,      # Input/output embedding dimension
    latent_dim=256,          # Shared latent space dimension
    hidden_dim=512,          # Hidden layer dimension
    adapter_depth=3,         # Adapter network depth
    backbone_depth=4,        # Backbone network depth
    dropout=0.1,             # Dropout rate
    use_conditioning=False,  # Enable conditioning
    cond_dim=0              # Conditioning dimension
)
```

### Training Configuration

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
    lambda_latent=1.0,       # Latent adversarial loss weight
    weight_decay=0.01        # Weight decay
)
```

### Training with Monitoring

```python
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_dir='checkpoints',
    save_every=10,           # Save checkpoint every N epochs
    early_stopping_patience=20  # Early stopping patience
)

# Access training history
train_losses = history['train_losses']
val_losses = history['val_losses']
```

## API Reference

### Core Classes

#### `NL2CMTranslator`

Main translation model class.

```python
class NL2CMTranslator(nn.Module):
    def __init__(self, embedding_dim, latent_dim=256, hidden_dim=512, 
                 adapter_depth=3, backbone_depth=4, dropout=0.1,
                 use_conditioning=False, cond_dim=0):
        """Initialize the NL2CM translator."""
        
    def forward(self, batch, condition=None):
        """Forward pass through the translator."""
        
    def translate_nlt_to_cmt(self, nlt_emb, condition=None):
        """Translate NL embeddings to CM embeddings."""
        
    def translate_cmt_to_nlt(self, cmt_emb, condition=None):
        """Translate CM embeddings to NL embeddings."""
```

#### `NL2CMTrainer`

Training orchestrator with all loss functions.

```python
class NL2CMTrainer:
    def __init__(self, model, device='cuda', lr_generator=1e-4, 
                 lr_discriminator=4e-4, **loss_weights):
        """Initialize the trainer."""
        
    def train_step(self, batch):
        """Perform one training step."""
        
    def validate(self, val_loader):
        """Validate the model."""
        
    def train(self, train_loader, val_loader, epochs, save_dir='checkpoints',
              save_every=10, early_stopping_patience=20):
        """Train the model."""
```

#### `NL2CMEvaluator`

Comprehensive evaluation with all vec2vec metrics.

```python
class NL2CMEvaluator:
    def __init__(self, model, device='cuda'):
        """Initialize the evaluator."""
        
    def evaluate_all(self, nlt_emb, cmt_emb, labels=None):
        """Compute all evaluation metrics."""
        
    def compute_cosine_similarity(self, nlt_emb, cmt_emb):
        """Compute mean cosine similarity."""
        
    def compute_top_k_accuracy(self, nlt_emb, cmt_emb, k=1):
        """Compute Top-K accuracy."""
        
    def compute_mean_rank(self, nlt_emb, cmt_emb):
        """Compute mean rank."""
```

### Data Classes

#### `NL2CMDataset`

Unpaired dataset for training.

```python
class NL2CMDataset(Dataset):
    def __init__(self, nlt_embeddings, cmt_embeddings, normalize=True, noise_level=0.0):
        """Initialize unpaired dataset."""
        
    def __getitem__(self, idx):
        """Get random unpaired sample."""
```

#### `PairedNL2CMDataset`

Paired dataset for evaluation.

```python
class PairedNL2CMDataset(Dataset):
    def __init__(self, nlt_embeddings, cmt_embeddings, normalize=True):
        """Initialize paired dataset."""
        
    def __getitem__(self, idx):
        """Get paired sample."""
```

## Evaluation Metrics

The library implements comprehensive evaluation metrics matching the vec2vec paper:

### Basic Translation Metrics

1. **Cosine Similarity**: Mean cosine similarity between translated and target embeddings
2. **Mean Rank**: Average rank of correct answers in retrieval
3. **Top-K Accuracy**: Fraction of queries where correct answer is in top-K
4. **MRR**: Mean Reciprocal Rank

### Advanced Metrics

1. **Cycle Consistency**: Round-trip translation fidelity
2. **Geometry Preservation**: Correlation of pairwise similarities
3. **Classification Performance**: Clustering-based evaluation using ARI and NMI

### Example Results

```
================================================================================
NL2CM Translation Evaluation Results
================================================================================

Metric                    NL2CM         Identity      Procrustes    Random       
---------------------------------------------------------------------------------
Cosine Similarity         0.3714        0.6591        0.8945        N/A          
Mean Rank                 488.67        N/A           N/A           489.00       
Top-1 Accuracy            0.0010        0.0000        0.0000        0.0000       
Top-5 Accuracy            0.0041        0.0000        0.0000        0.0000       
MRR                       0.0075        0.0000        0.0000        0.0000       

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

## Examples

### Example 1: Basic Training and Evaluation

```python
import torch
from nl2cm import NL2CMTranslator, NL2CMTrainer, NL2CMEvaluator, load_nl2cm_data

# Load data
train_loader, val_loader, test_loader = load_nl2cm_data(
    'datasets/eamodelset_nl2cm_embeddings_df.pkl'
)

# Create and train model
model = NL2CMTranslator(embedding_dim=1536, latent_dim=256)
trainer = NL2CMTrainer(model, device='cuda')

print("Starting training...")
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    save_dir='checkpoints'
)

# Evaluate
evaluator = NL2CMEvaluator(model, device='cuda')
results = evaluator.evaluate_all(nlt_eval, cmt_eval)
print(evaluator.create_evaluation_table(results))
```

### Example 2: Custom Model Configuration

```python
# Custom architecture
model = NL2CMTranslator(
    embedding_dim=1536,
    latent_dim=512,          # Larger latent space
    hidden_dim=1024,         # Deeper networks
    adapter_depth=4,         # More adapter layers
    backbone_depth=6,        # Deeper backbone
    dropout=0.2,             # Higher dropout
    use_conditioning=True,   # Enable conditioning
    cond_dim=32              # Conditioning dimension
)

# Custom training configuration
trainer = NL2CMTrainer(
    model=model,
    lr_generator=5e-5,       # Lower learning rate
    lr_discriminator=2e-4,   # Adjusted discriminator LR
    lambda_rec=20.0,         # Higher reconstruction weight
    lambda_cyc=20.0,         # Higher cycle consistency weight
    lambda_vsp=5.0,          # Higher VSP weight
    weight_decay=0.005       # Lower weight decay
)
```

### Example 3: Batch Translation

```python
# Translate multiple embeddings
def batch_translate(model, nlt_embeddings, batch_size=32):
    """Translate embeddings in batches."""
    model.eval()
    translated = []
    
    with torch.no_grad():
        for i in range(0, len(nlt_embeddings), batch_size):
            batch = nlt_embeddings[i:i+batch_size]
            batch_translated = model.translate_nlt_to_cmt(batch)
            translated.append(batch_translated)
    
    return torch.cat(translated, dim=0)

# Usage
nlt_embeddings = torch.randn(1000, 1536)  # 1000 NL embeddings
cm_embeddings = batch_translate(model, nlt_embeddings)
print(f"Translated {len(cm_embeddings)} embeddings")
```

### Example 4: Evaluation with Custom Metrics

```python
# Custom evaluation
def evaluate_translation_quality(model, nlt_emb, cmt_emb):
    """Custom evaluation function."""
    evaluator = NL2CMEvaluator(model)
    
    # Basic metrics
    cosine_sim = evaluator.compute_cosine_similarity(nlt_emb, cmt_emb)
    top1_acc = evaluator.compute_top_k_accuracy(nlt_emb, cmt_emb, k=1)
    mean_rank = evaluator.compute_mean_rank(nlt_emb, cmt_emb)
    
    # Cycle consistency
    cycle_metrics = evaluator.compute_cycle_consistency_metrics(nlt_emb, cmt_emb)
    
    # Geometry preservation
    geometry_metrics = evaluator.compute_geometry_preservation_metrics(nlt_emb, cmt_emb)
    
    return {
        'cosine_similarity': cosine_sim,
        'top1_accuracy': top1_acc,
        'mean_rank': mean_rank,
        'cycle_consistency': cycle_metrics['mean_cycle_similarity'],
        'geometry_preservation': geometry_metrics['mean_geometry_correlation']
    }

# Usage
results = evaluate_translation_quality(model, nlt_eval, cmt_eval)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

## Advanced Features

### 1. Conditioning Support

Enable conditioning on target modeling language:

```python
# Model with conditioning
model = NL2CMTranslator(
    embedding_dim=1536,
    latent_dim=256,
    use_conditioning=True,
    cond_dim=32
)

# Translation with condition
condition = torch.randn(batch_size, 32)  # Target language embedding
cm_embeddings = model.translate_nlt_to_cmt(nlt_embeddings, condition)
```

### 2. Custom Loss Functions

```python
class CustomNL2CMTrainer(NL2CMTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss_weight = kwargs.get('lambda_custom', 1.0)
    
    def compute_generator_loss(self, outputs, batch):
        # Get standard losses
        losses = super().compute_generator_loss(outputs, batch)
        
        # Add custom loss
        custom_loss = self.compute_custom_loss(outputs, batch)
        losses['total'] += self.custom_loss_weight * custom_loss
        losses['custom'] = custom_loss
        
        return losses
    
    def compute_custom_loss(self, outputs, batch):
        # Implement your custom loss here
        return torch.tensor(0.0, device=self.device)
```

### 3. Multi-GPU Training

```python
import torch.nn as nn

# Wrap model for multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

trainer = NL2CMTrainer(model, device='cuda')
```

### 4. Custom Data Loading

```python
from torch.utils.data import Dataset, DataLoader

class CustomNL2CMDataset(Dataset):
    def __init__(self, nlt_embeddings, cmt_embeddings, custom_features=None):
        self.nlt_embeddings = nlt_embeddings
        self.cmt_embeddings = cmt_embeddings
        self.custom_features = custom_features
    
    def __len__(self):
        return len(self.nlt_embeddings)
    
    def __getitem__(self, idx):
        sample = {
            'nlt': torch.FloatTensor(self.nlt_embeddings[idx]),
            'cmt': torch.FloatTensor(self.cmt_embeddings[idx])
        }
        
        if self.custom_features is not None:
            sample['custom'] = torch.FloatTensor(self.custom_features[idx])
        
        return sample

# Usage
custom_dataset = CustomNL2CMDataset(nlt_emb, cmt_emb, custom_features)
custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   train_loader = DataLoader(dataset, batch_size=16)  # Instead of 32
   
   # Use gradient accumulation
   trainer = NL2CMTrainer(model, device='cuda')
   # Implement gradient accumulation in training loop
   ```

2. **Training Instability**
   ```python
   # Reduce learning rates
   trainer = NL2CMTrainer(
       model=model,
       lr_generator=5e-5,      # Lower generator LR
       lr_discriminator=2e-4   # Lower discriminator LR
   )
   
   # Adjust loss weights
   trainer = NL2CMTrainer(
       model=model,
       lambda_rec=10.0,        # Lower reconstruction weight
       lambda_cyc=10.0,        # Lower cycle consistency weight
       lambda_vsp=1.0          # Lower VSP weight
   )
   ```

3. **Poor Translation Quality**
   ```python
   # Increase model capacity
   model = NL2CMTranslator(
       embedding_dim=1536,
       latent_dim=512,         # Larger latent space
       hidden_dim=1024,        # Larger hidden layers
       adapter_depth=4,        # Deeper adapters
       backbone_depth=6        # Deeper backbone
   )
   
   # Increase training time
   history = trainer.train(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=200,             # More epochs
       early_stopping_patience=50  # More patience
   )
   ```

4. **Data Loading Issues**
   ```python
   # Check data format
   import pickle
   with open('data.pkl', 'rb') as f:
       df = pickle.load(f)
   
   print(df.columns)  # Should include 'NL_Serialization_Emb', 'CM_Serialization_Emb'
   print(df['NL_Serialization_Emb'].iloc[0].shape)  # Should be (embedding_dim,)
   ```

### Performance Tips

1. **Memory Optimization**
   - Use smaller batch sizes for large models
   - Enable gradient checkpointing for very deep networks
   - Use mixed precision training

2. **Training Speed**
   - Use multiple workers for data loading
   - Enable CUDA optimizations
   - Use gradient accumulation for effective larger batch sizes

3. **Model Selection**
   - Start with default hyperparameters
   - Increase model capacity gradually
   - Monitor validation loss for overfitting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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

## Acknowledgments

- Based on the vec2vec approach from Jha et al. (2025)
- Implements adversarial training and geometry preservation
- Extends to NL2CM translation task
