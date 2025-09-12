"""
Test NL2CM Pipeline

This script tests the complete NL2CM pipeline on the provided embeddings.
It demonstrates the entire workflow from data loading to evaluation.
"""

import json
from nl2cm.evaluation import NL2CMEvaluator
from nl2cm.training import NL2CMTrainer
from nl2cm.model import NL2CMTranslator
from nl2cm.data_loader import create_evaluation_splits, load_nl2cm_data
import sys
import torch
from pathlib import Path
import os
from torch.utils.data import DataLoader, TensorDataset

from nl2cm.utils import get_device
from nl2cm.parse_args import parse_args

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))


def test_data_loading(data_path, nl_cm_cols):
    """Test data loading functionality."""
    print("=" * 60)
    print("Testing Data Loading")
    print("=" * 60)

    # Test data loaders
    data = load_nl2cm_data(data_path, nl_cm_cols, test_size=0.2)
    embedding_dim = data['embedding_dim']
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"NL batch shape: {batch['nlt'].shape}")
    print(f"CM batch shape: {batch['cmt'].shape}")

    print("✓ Data loading test passed\n")
    return embedding_dim


def test_model_creation(embedding_dim):
    """Test model creation and forward pass."""
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)

    # Create model
    model = NL2CMTranslator(
        embedding_dim=embedding_dim,
        latent_dim=256,
        hidden_dim=512,
        adapter_depth=3,
        backbone_depth=4,
        dropout=0.1
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    device = get_device()
    model = model.to(device)

    # Create dummy batch
    batch_size = 4
    batch = {
        'nlt': torch.randn(batch_size, embedding_dim).to(device),
        'cmt': torch.randn(batch_size, embedding_dim).to(device)
    }

    # Forward pass
    with torch.no_grad():
        outputs = model(batch)

    print(f"Output keys: {outputs.keys()}")
    print(f"NL to CM shape: {outputs['nlt_to_cmt'].shape}")
    print(f"CM to NL shape: {outputs['cmt_to_nlt'].shape}")
    print(f"NL recon shape: {outputs['nlt_recon'].shape}")
    print(f"CM recon shape: {outputs['cmt_recon'].shape}")

    # Test translation methods
    nlt_translated = model.translate_nlt_to_cmt(batch['nlt'])
    cmt_translated = model.translate_cmt_to_nlt(batch['cmt'])

    print(f"Translation NL to CM shape: {nlt_translated.shape}")
    print(f"Translation CM to NL shape: {cmt_translated.shape}")

    print("✓ Model creation test passed\n")
    return model, device


def test_training(model, embedding_dim):
    """Test training functionality."""
    print("=" * 60)
    print("Testing Training")
    print("=" * 60)

    # Create trainer with TensorBoard logging
    trainer = NL2CMTrainer(
        model=model,
        lr_generator=1e-4,
        lr_discriminator=4e-4,
        lambda_rec=15.0,
        lambda_cyc=15.0,
        lambda_vsp=2.0,
        lambda_adv=1.0,
        lambda_latent=1.0,
        use_tensorboard=True,
        log_dir='test_tensorboard_logs'
    )

    # Create dummy data loaders

    nlt_data = torch.randn(100, embedding_dim)
    cmt_data = torch.randn(100, embedding_dim)

    train_dataset = TensorDataset(nlt_data, cmt_data)
    val_dataset = TensorDataset(nlt_data[:20], cmt_data[:20])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Test training step
    batch = next(iter(train_loader))
    batch = {'nlt': batch[0], 'cmt': batch[1]}

    losses = trainer.train_step(batch)
    print(f"Training step losses: {losses}")

    # Test validation
    val_losses = trainer.validate(val_loader)
    print(f"Validation losses: {val_losses}")

    # Close TensorBoard logger
    trainer.close_tensorboard()

    print("✓ Training test passed\n")
    return trainer


def test_evaluation(model, embedding_dim):
    """Test evaluation functionality."""
    print("=" * 60)
    print("Testing Evaluation")
    print("=" * 60)
    
    device = get_device()
    # Create evaluator
    evaluator = NL2CMEvaluator(model)

    # Create dummy evaluation data
    nlt_eval = torch.randn(50, embedding_dim).to(device)
    cmt_eval = torch.randn(50, embedding_dim).to(device)

    # Test individual metrics
    cosine_sim = evaluator.compute_cosine_similarity(nlt_eval, cmt_eval)
    print(f"Cosine similarity: {cosine_sim:.4f}")

    top1_acc = evaluator.compute_top_k_accuracy(nlt_eval, cmt_eval, k=1)
    print(f"Top-1 accuracy: {top1_acc:.4f}")

    mean_rank = evaluator.compute_mean_rank(nlt_eval, cmt_eval)
    print(f"Mean rank: {mean_rank:.2f}")

    # Test comprehensive evaluation
    results = evaluator.evaluate(nlt_eval, cmt_eval)
    print(f"Evaluation results keys: {list(results.keys())}")

    # Test evaluation table
    table = evaluator.create_evaluation_table(results)
    print("Evaluation table created successfully")

    print("✓ Evaluation test passed\n")
    return evaluator


def test_full_pipeline():
    """Test the complete pipeline on real data."""
    
    print("=" * 60)
    print("Testing Full Pipeline on Real Data")
    print("=" * 60)
    args = parse_args()
    data_path = os.path.join(args.data_path, args.dataset)
    nl_cm_cols = [args.nl_col, args.cm_col]
    
    # Load real data
    data = load_nl2cm_data(data_path, nl_cm_cols, test_size=0.2, limit=args.limit, batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    embedding_dim = data['embedding_dim']

    # Create model
    model = NL2CMTranslator(
        embedding_dim=embedding_dim,
        latent_dim=256,
        hidden_dim=512,
        adapter_depth=3,
        backbone_depth=4,
        dropout=0.1
    )

    device = get_device()
    model = model.to(device)

    # Create trainer with TensorBoard logging
    trainer = NL2CMTrainer(
        model=model,
        lr_generator=1e-4,
        lr_discriminator=4e-4,
        lambda_rec=15.0,
        lambda_cyc=15.0,
        lambda_vsp=2.0,
        lambda_adv=1.0,
        lambda_latent=1.0,
        use_tensorboard=True,
        log_dir=args.output_dir
    )

    # Train for a few epochs
    print(f"Training for {args.epochs} epochs...")
    save_dir = os.path.join(args.output_dir, args.save_dir)
    eval_dir = os.path.join(args.output_dir, args.eval_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=save_dir,
        save_every=args.save_every
    )

    # Evaluate on test data
    print("Evaluating on test data...")
    evaluator = NL2CMEvaluator(
        model=model,
        use_tensorboard=True,
        tensorboard_logger=trainer.tensorboard_logger
    )

    os.makedirs(eval_dir, exist_ok=True)
    results = evaluator.evaluate_data_loader(data['test_loader'])
    with open(os.path.join(eval_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
    print(f"Evaluation results saved to {os.path.join(eval_dir, 'results.json')}")        
    print("\n" + evaluator.create_evaluation_table(results))

    # Close TensorBoard logger
    trainer.close_tensorboard()

    print("✓ Full pipeline test passed\n")
    

def main():
    """Run all tests."""
    print("Starting NL2CM Pipeline Tests")
    print("=" * 80)
    
    try:
        test_full_pipeline()

        print("=" * 80)
        print("All tests passed successfully! ✓")
        print("=" * 80)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
