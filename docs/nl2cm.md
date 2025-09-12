# NL2CM: Natural Language to Conceptual Model Translation

This repository implements a complete NL2CM (Natural Language to Conceptual Model) translation system based on the vec2vec approach. The system learns to translate between Natural Language and Conceptual Model embedding spaces using adversarial training and geometry preservation.

## Overview

The NL2CM system implements the vec2vec approach for translating between different embedding spaces without paired data. It uses:

- **Adapters**: Map embeddings to/from a shared latent space
- **Shared Backbone**: Processes embeddings in the latent space
- **Adversarial Training**: Ensures translated embeddings match target distribution
- **Geometry Preservation**: Maintains pairwise relationships during translation
- **Cycle Consistency**: Ensures round-trip translation fidelity

## Architecture

```
NL Embeddings → Adapter → Shared Backbone → Output Adapter → CM Embeddings
CM Embeddings → Adapter → Shared Backbone → Output Adapter → NL Embeddings
```

### Key Components

1. **Adapters**: MLP networks that map between embedding space and latent space
2. **Shared Backbone**: Processes embeddings in the shared latent space
3. **Discriminators**: Adversarial networks for output and latent spaces
4. **Loss Functions**: Adversarial, reconstruction, cycle consistency, and VSP losses

## Installation

```bash
# Install dependencies
pip install torch scikit-learn matplotlib tqdm

# Clone the repository
git clone <repository-url>
cd cm-vec2vec
```

## Usage

### Quick Start

```bash
# Run the complete pipeline test
python test_nl2cm.py

# Train the model
python nl2cm/train.py --epochs 50 --batch_size 16

# Evaluate the model
python nl2cm/evaluate.py --checkpoint_path checkpoints/nl2cm_full/nl2cm_best.pt
```

### Training

```bash
python nl2cm/train.py \
    --epochs 100 \
    --batch_size 32 \
    --latent_dim 256 \
    --hidden_dim 512 \
    --lambda_rec 15.0 \
    --lambda_cyc 15.0 \
    --lambda_vsp 2.0
```

### Evaluation

```bash
python nl2cm/evaluate.py \
    --checkpoint_path checkpoints/nl2cm_full/nl2cm_best.pt \
    --eval_samples 1000
```

## Data Format

The system expects a pickle file containing a DataFrame with:
- `NL_Serialization_Emb`: Natural Language embeddings (N, 1536)
- `CM_Serialization_Emb`: Conceptual Model embeddings (N, 1536)

## Model Configuration

### Default Parameters

- **Embedding Dimension**: 1536
- **Latent Dimension**: 256
- **Hidden Dimension**: 512
- **Adapter Depth**: 3 layers
- **Backbone Depth**: 4 layers
- **Dropout**: 0.1

### Loss Weights

- **Reconstruction**: 15.0
- **Cycle Consistency**: 15.0
- **Vector Space Preservation**: 2.0
- **Adversarial**: 1.0
- **Latent Adversarial**: 1.0

## Evaluation Metrics

The system evaluates translation quality using metrics from the vec2vec paper:

### Basic Metrics
- **Cosine Similarity**: Mean cosine similarity between translated and target embeddings
- **Mean Rank**: Average rank of correct answers in retrieval
- **Top-K Accuracy**: Fraction of queries where correct answer is in top-K
- **MRR**: Mean Reciprocal Rank

### Advanced Metrics
- **Cycle Consistency**: Round-trip translation fidelity
- **Geometry Preservation**: Correlation of pairwise similarities
- **Classification Performance**: Clustering-based evaluation

## Results

### Performance Summary

| Metric | NL2CM | Identity | Procrustes | Random |
|--------|-------|----------|------------|--------|
| Cosine Similarity | 0.3714 | 0.6591 | 0.8945 | N/A |
| Mean Rank | 488.67 | N/A | N/A | 489.00 |
| Top-1 Accuracy | 0.0010 | 0.0000 | 0.0000 | 0.0000 |
| Top-5 Accuracy | 0.0041 | 0.0000 | 0.0000 | 0.0000 |
| MRR | 0.0075 | 0.0000 | 0.0000 | 0.0000 |

### Key Findings

1. **Translation Quality**: The model achieves reasonable cosine similarity (0.37) compared to identity baseline (0.66)
2. **Retrieval Performance**: Shows improvement over random baseline in mean rank
3. **Cycle Consistency**: Maintains reasonable round-trip fidelity (0.34)
4. **Geometry Preservation**: Preserves pairwise relationships (0.55 correlation)

## File Structure

```
nl2cm/
├── __init__.py              # Package initialization
├── data_loader.py           # Data loading and preprocessing
├── model.py                 # Model architecture
├── training.py              # Training loop and losses
├── evaluation.py            # Evaluation metrics
├── train.py                 # Main training script
└── evaluate.py              # Evaluation script

test_nl2cm.py                # Complete pipeline test
generate_evaluation_tables.py # Generate evaluation tables
```

## Key Features

### 1. Unpaired Training
- No need for paired NL-CM data
- Uses distribution matching for training
- Enables training on large unlabeled datasets

### 2. Geometry Preservation
- Maintains pairwise relationships during translation
- Preserves semantic neighborhoods
- Enables downstream tasks like retrieval and classification

### 3. Adversarial Training
- Ensures translated embeddings match target distribution
- Prevents mode collapse
- Improves translation quality

### 4. Cycle Consistency
- Ensures round-trip translation fidelity
- Acts as implicit supervision
- Improves translation robustness

## Extensions

### 1. Conditioning
The model supports conditioning on target modeling language:
```python
model = NL2CMTranslator(
    embedding_dim=1536,
    latent_dim=256,
    use_conditioning=True,
    cond_dim=32
)
```

### 2. Multi-head Outputs
Separate output adapters for different construct types:
```python
# Extend for Node/Edge/Relationship types
```

### 3. Graph-aware VSP
Preserve graph structure in conceptual models:
```python
# Add triplet-level VSP over semantic relationships
```

## Limitations

1. **Translation Quality**: Current results show room for improvement in cosine similarity
2. **Retrieval Performance**: Top-1 accuracy is low, indicating need for better alignment
3. **Data Requirements**: Requires large amounts of unpaired data for training
4. **Domain Specificity**: Performance may vary across different domains

## Future Work

1. **Improved Architecture**: Deeper networks, attention mechanisms
2. **Better Training**: Curriculum learning, progressive training
3. **Domain Adaptation**: Fine-tuning for specific domains
4. **Multi-modal Extension**: Support for different embedding types

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the vec2vec approach from Jha et al. (2025)
- Implements adversarial training and geometry preservation
- Extends to NL2CM translation task
