# NL2CM: Natural Language to Conceptual Model Translation

A PyTorch implementation of the vec2vec approach for translating between Natural Language (NL) and Conceptual Model (CM) embedding spaces. This package enables unsupervised translation between different embedding domains using adversarial training and geometry preservation.

## Overview

The NL2CM system implements the vec2vec methodology for translating between embedding spaces without paired data. It uses a modular architecture with adapters, shared backbone, and adversarial training to learn meaningful translations between Natural Language and Conceptual Model representations.

### Key Features

- **Unpaired Training**: No need for paired NL-CM data
- **Adversarial Learning**: Output and latent space discriminators
- **Geometry Preservation**: Maintains pairwise relationships during translation
- **Cycle Consistency**: Ensures round-trip translation fidelity
- **Modular Design**: Easy to extend and customize
- **Comprehensive Evaluation**: All metrics from the vec2vec paper
- **TensorBoard Integration**: Real-time monitoring of training and evaluation metrics

## Architecture

```
NL Embeddings → Adapter → Shared Backbone → Output Adapter → CM Embeddings
CM Embeddings → Adapter → Shared Backbone → Output Adapter → NL Embeddings
```

### Components

1. **Adapters**: MLP networks that map between embedding space and latent space
2. **Shared Backbone**: Processes embeddings in the shared latent space
3. **Discriminators**: Adversarial networks for output and latent spaces
4. **Loss Functions**: Adversarial, reconstruction, cycle consistency, and VSP losses

## Installation

```bash
# Install dependencies
pip install torch scikit-learn matplotlib tqdm tensorboardX

# Clone the repository
git clone <repository-url>
cd cm-vec2vec
```

## Quick Start

### Basic Usage

```python
from nl2cm import NL2CMTranslator, NL2CMTrainer, NL2CMEvaluator
from nl2cm.data_loader import load_nl2cm_data

# Load data
train_loader, val_loader, test_loader = load_nl2cm_data('path/to/data.pkl')

# Create model
model = NL2CMTranslator(
    embedding_dim=1536,
    latent_dim=256,
    hidden_dim=512
)

# Train model
trainer = NL2CMTrainer(model, device='cuda')
history = trainer.train(train_loader, val_loader, epochs=50)

# Evaluate model
evaluator = NL2CMEvaluator(model, device='cuda')
results = evaluator.evaluate_all(nlt_embeddings, cmt_embeddings)
```

### Command Line Interface

```bash
# Train the model with TensorBoard logging
python nl2cm/train.py --epochs 50 --batch_size 16 --use_tensorboard

# Evaluate the model
python nl2cm/evaluate.py --checkpoint_path checkpoints/nl2cm_best.pt

# Run complete pipeline test
python test_nl2cm.py

# Run TensorBoard demo
python run_tensorboard_demo.py
```

## Package Structure

```
nl2cm/
├── __init__.py              # Package initialization
├── data_loader.py           # Data loading and preprocessing
├── model.py                 # Model architecture
├── training.py              # Training loop and losses
├── evaluation.py            # Evaluation metrics
├── tensorboard_logger.py    # TensorBoard logging utilities
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
└── README.md                # This file
```

## API Reference

### Data Loading

#### `load_nl2cm_data(data_path, test_size=0.2, random_state=42)`

Load NL2CM data and create train/validation/test splits.

**Parameters:**
- `data_path` (str): Path to the pickle file containing embeddings
- `test_size` (float): Fraction of data to use for testing
- `random_state` (int): Random state for reproducibility

**Returns:**
- `Tuple[DataLoader, DataLoader, DataLoader]`: Train, validation, and test loaders

#### `create_evaluation_splits(data_path, n_eval_samples=1000)`

Create evaluation splits for computing vec2vec-style metrics.

**Parameters:**
- `data_path` (str): Path to the pickle file containing embeddings
- `n_eval_samples` (int): Number of samples to use for evaluation

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: NL and CM evaluation embeddings

### Model Architecture

#### `NL2CMTranslator`

Main translation model implementing the vec2vec approach.

**Parameters:**
- `embedding_dim` (int): Dimension of input embeddings (default: 1536)
- `latent_dim` (int): Dimension of latent space (default: 256)
- `hidden_dim` (int): Dimension of hidden layers (default: 512)
- `adapter_depth` (int): Number of layers in adapters (default: 3)
- `backbone_depth` (int): Number of layers in backbone (default: 4)
- `dropout` (float): Dropout rate (default: 0.1)
- `use_conditioning` (bool): Whether to use conditioning (default: False)
- `cond_dim` (int): Dimension of conditioning vector (default: 0)

**Methods:**
- `forward(batch, condition=None)`: Forward pass through the translator
- `translate_nlt_to_cmt(nlt_emb, condition=None)`: Translate NL to CM embeddings
- `translate_cmt_to_nlt(cmt_emb, condition=None)`: Translate CM to NL embeddings

### Training

#### `NL2CMTrainer`

Trainer for the NL2CM translation model.

**Parameters:**
- `model` (NL2CMTranslator): The translation model
- `device` (str): Device to run training on (default: 'cuda')
- `lr_generator` (float): Learning rate for generator (default: 1e-4)
- `lr_discriminator` (float): Learning rate for discriminator (default: 4e-4)
- `lambda_rec` (float): Weight for reconstruction loss (default: 15.0)
- `lambda_cyc` (float): Weight for cycle consistency loss (default: 15.0)
- `lambda_vsp` (float): Weight for vector space preservation loss (default: 2.0)
- `lambda_adv` (float): Weight for adversarial loss (default: 1.0)
- `lambda_latent` (float): Weight for latent adversarial loss (default: 1.0)
- `weight_decay` (float): Weight decay for optimizers (default: 0.01)

**Methods:**
- `train(train_loader, val_loader, epochs, save_dir, save_every, early_stopping_patience)`: Train the model
- `validate(val_loader)`: Validate the model
- `load_checkpoint(checkpoint_path)`: Load a checkpoint

### Evaluation

#### `NL2CMEvaluator`

Evaluator for the NL2CM translation model.

**Parameters:**
- `model` (NL2CMTranslator): The trained model
- `device` (str): Device to run evaluation on (default: 'cuda')

**Methods:**
- `evaluate_all(nlt_emb, cmt_emb, labels=None)`: Compute all evaluation metrics
- `compute_cosine_similarity(nlt_emb, cmt_emb)`: Compute mean cosine similarity
- `compute_top_k_accuracy(nlt_emb, cmt_emb, k=1)`: Compute Top-K accuracy
- `compute_mean_rank(nlt_emb, cmt_emb)`: Compute mean rank
- `create_evaluation_table(results)`: Create formatted evaluation table

## Data Format

The system expects a pickle file containing a DataFrame with the following columns:

- `NL_Serialization_Emb`: Natural Language embeddings (N, 1536)
- `CM_Serialization_Emb`: Conceptual Model embeddings (N, 1536)

Example:
```python
import pickle
import pandas as pd

# Load data
with open('data.pkl', 'rb') as f:
    df = pickle.load(f)

# Extract embeddings
nlt_embeddings = np.stack(df['NL_Serialization_Emb'].values)
cmt_embeddings = np.stack(df['CM_Serialization_Emb'].values)
```

## Training Configuration

### Default Parameters

```python
# Model parameters
embedding_dim = 1536
latent_dim = 256
hidden_dim = 512
adapter_depth = 3
backbone_depth = 4
dropout = 0.1

# Training parameters
epochs = 50
batch_size = 16
lr_generator = 1e-4
lr_discriminator = 4e-4
weight_decay = 0.01

# Loss weights
lambda_rec = 15.0      # Reconstruction
lambda_cyc = 15.0      # Cycle consistency
lambda_vsp = 2.0       # Vector space preservation
lambda_adv = 1.0       # Adversarial
lambda_latent = 1.0    # Latent adversarial
```

### Training Script

```bash
python nl2cm/train.py \
    --epochs 100 \
    --batch_size 32 \
    --latent_dim 256 \
    --hidden_dim 512 \
    --lambda_rec 15.0 \
    --lambda_cyc 15.0 \
    --lambda_vsp 2.0 \
    --save_dir checkpoints/nl2cm
```

## Evaluation Metrics

The system evaluates translation quality using comprehensive metrics:

### Basic Metrics
- **Cosine Similarity**: Mean cosine similarity between translated and target embeddings
- **Mean Rank**: Average rank of correct answers in retrieval
- **Top-K Accuracy**: Fraction of queries where correct answer is in top-K
- **MRR**: Mean Reciprocal Rank

### Advanced Metrics
- **Cycle Consistency**: Round-trip translation fidelity
- **Geometry Preservation**: Correlation of pairwise similarities
- **Classification Performance**: Clustering-based evaluation

### Example Results

| Metric | NL2CM | Identity | Procrustes | Random |
|--------|-------|----------|------------|--------|
| Cosine Similarity | 0.3714 | 0.6591 | 0.8945 | N/A |
| Mean Rank | 488.67 | N/A | N/A | 489.00 |
| Top-1 Accuracy | 0.0010 | 0.0000 | 0.0000 | 0.0000 |
| Top-5 Accuracy | 0.0041 | 0.0000 | 0.0000 | 0.0000 |
| MRR | 0.0075 | 0.0000 | 0.0000 | 0.0000 |

## Loss Functions

The training objective combines multiple loss functions:

### 1. Adversarial Loss
Ensures translated embeddings match target distribution:
```python
L_adv = L_GAN(D_C, F_N2C) + L_GAN(D_N, F_C2N) + L_GAN(D_latent, T)
```

### 2. Reconstruction Loss
Maintains identity within each space:
```python
L_rec = ||R_N(x) - x||² + ||R_C(y) - y||²
```

### 3. Cycle Consistency Loss
Ensures round-trip translation fidelity:
```python
L_cyc = ||F_C2N(F_N2C(x)) - x||² + ||F_N2C(F_C2N(y)) - y||²
```

### 4. Vector Space Preservation Loss
Preserves pairwise relationships:
```python
L_vsp = ||x_i·x_j - F_N2C(x_i)·F_N2C(x_j)||² + ||y_i·y_j - F_C2N(y_i)·F_C2N(y_j)||²
```

## TensorBoard Integration

The NL2CM package includes comprehensive TensorBoard logging for monitoring training progress and evaluation metrics.

### Features

- **Training Losses**: Real-time tracking of all loss components
- **Validation Metrics**: Monitor model performance during training
- **Learning Rates**: Track optimizer learning rate schedules
- **Model Parameters**: Histograms of weights and gradients
- **Evaluation Metrics**: Comprehensive evaluation results
- **Translation Examples**: Visualize translation quality improvements
- **Embedding Visualizations**: PCA-reduced embedding plots

### Usage

#### Basic TensorBoard Logging

```python
from nl2cm import NL2CMTrainer, NL2CMEvaluator

# Create trainer with TensorBoard logging
trainer = NL2CMTrainer(
    model=model,
    device='cuda',
    use_tensorboard=True,
    log_dir='tensorboard_logs'
)

# Create evaluator with TensorBoard logging
evaluator = NL2CMEvaluator(
    model=model,
    device='cuda',
    use_tensorboard=True,
    tensorboard_logger=trainer.tensorboard_logger
)

# Train with logging
history = trainer.train(train_loader, val_loader, epochs=50)

# Evaluate with logging
results = evaluator.evaluate_all(nlt_embeddings, cmt_embeddings)

# Close logger
trainer.close_tensorboard()
```

#### Command Line with TensorBoard

```bash
# Train with TensorBoard logging
python nl2cm/train.py --epochs 50 --use_tensorboard --tensorboard_dir logs

# View logs
tensorboard --logdir logs
```

#### Custom TensorBoard Logger

```python
from nl2cm import create_tensorboard_logger

# Create custom logger
logger = create_tensorboard_logger(
    log_dir='experiments',
    experiment_name='nl2cm_experiment_1'
)

# Use with trainer
trainer = NL2CMTrainer(
    model=model,
    device='cuda',
    use_tensorboard=True,
    log_dir='experiments'
)

# Log custom metrics
logger.log_text("Experiment started", "status")
logger.log_hyperparameters(hparams, metrics)
```

### Viewing Results

1. **Start TensorBoard**:
   ```bash
   tensorboard --logdir tensorboard_logs
   ```

2. **Open Browser**: Navigate to `http://localhost:6006`

3. **Available Tabs**:
   - **SCALARS**: Loss curves, metrics, learning rates
   - **HISTOGRAMS**: Model parameter distributions
   - **IMAGES**: Embedding visualizations
   - **TEXT**: Logged text and hyperparameters

### Logged Metrics

#### Training Metrics
- `Training/Total_Generator_Loss`
- `Training/Total_Discriminator_Loss`
- `Training/Reconstruction_Loss`
- `Training/Cycle_Consistency_Loss`
- `Training/VSP_Loss`
- `Training/Adversarial_Loss`
- `Training/Latent_Adversarial_Loss`

#### Validation Metrics
- `Validation/Total_Generator_Loss`
- `Validation/Total_Discriminator_Loss`
- All individual loss components

#### Evaluation Metrics
- `Final_Evaluation/Basic/cosine_similarity`
- `Final_Evaluation/Basic/mean_rank`
- `Final_Evaluation/Basic/top_1_accuracy`
- `Final_Evaluation/Cycle_Consistency/mean_cycle_similarity`
- `Final_Evaluation/Geometry_Preservation/mean_geometry_correlation`

#### Model Metrics
- `Learning_Rate/Generator`
- `Learning_Rate/Discriminator`
- `Model/Gradient_Norm`
- Parameter histograms for all layers

## Advanced Usage

### Conditioning

The model supports conditioning on target modeling language:

```python
model = NL2CMTranslator(
    embedding_dim=1536,
    latent_dim=256,
    use_conditioning=True,
    cond_dim=32
)

# Use conditioning during translation
condition = torch.randn(batch_size, 32)  # Target language encoding
translated = model.translate_nlt_to_cmt(nlt_emb, condition)
```

### Custom Datasets

Create custom datasets for different embedding types:

```python
from nl2cm.data_loader import NL2CMDataset

# Create custom dataset
dataset = NL2CMDataset(
    nlt_embeddings=your_nl_embeddings,
    cmt_embeddings=your_cm_embeddings,
    normalize=True,
    noise_level=0.1
)
```

### Model Checkpointing

Save and load model checkpoints:

```python
# Save checkpoint
trainer.save_checkpoint('model_checkpoint.pt')

# Load checkpoint
trainer.load_checkpoint('model_checkpoint.pt')
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 8`
   - Use gradient accumulation
   - Reduce model dimensions

2. **Training Instability**
   - Adjust learning rates
   - Modify loss weights
   - Use gradient clipping

3. **Poor Translation Quality**
   - Increase training epochs
   - Tune hyperparameters
   - Check data quality

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Batch Size**: Use largest batch size that fits in memory
3. **Learning Rates**: Start with default values and adjust as needed
4. **Loss Weights**: Balance different loss components

## Examples

### Complete Training Pipeline with TensorBoard

```python
from nl2cm import NL2CMTranslator, NL2CMTrainer, NL2CMEvaluator
from nl2cm.data_loader import load_nl2cm_data

# Load data
train_loader, val_loader, test_loader = load_nl2cm_data('data.pkl')

# Create model
model = NL2CMTranslator(embedding_dim=1536, latent_dim=256)

# Create trainer with TensorBoard logging
trainer = NL2CMTrainer(
    model=model, 
    device='cuda',
    use_tensorboard=True,
    log_dir='experiments'
)

# Create evaluator with TensorBoard logging
evaluator = NL2CMEvaluator(
    model=model,
    device='cuda',
    use_tensorboard=True,
    tensorboard_logger=trainer.tensorboard_logger
)

# Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    save_dir='checkpoints'
)

# Evaluate model
results = evaluator.evaluate_all(nlt_embeddings, cmt_embeddings)

# Print results
print(evaluator.create_evaluation_table(results))

# Close TensorBoard logger
trainer.close_tensorboard()

# View results in TensorBoard
print("To view TensorBoard logs, run:")
print("tensorboard --logdir experiments")
```

### Custom Evaluation

```python
# Load trained model
model = NL2CMTranslator(embedding_dim=1536, latent_dim=256)
model.load_state_dict(torch.load('checkpoint.pt'))

# Translate embeddings
with torch.no_grad():
    translated = model.translate_nlt_to_cmt(nlt_embeddings)

# Compute custom metrics
similarity = F.cosine_similarity(translated, cmt_embeddings, dim=1).mean()
print(f"Translation similarity: {similarity:.4f}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code, please cite the original vec2vec paper:

```bibtex
@article{jha2025harnessing,
  title={Harnessing the Universal Geometry of Embeddings},
  author={Jha, Akshita and Zhang, Yuchen and Shmatikov, Vitaly and Morris, Christopher},
  journal={arXiv preprint arXiv:2505.12540},
  year={2025}
}
```

## Acknowledgments

- Based on the vec2vec approach from Jha et al. (2025)
- Implements adversarial training and geometry preservation
- Extends to NL2CM translation task
- Built with PyTorch and modern deep learning practices
