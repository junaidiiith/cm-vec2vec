"""
Test NL2CM Pipeline

This script tests the complete NL2CM pipeline on the provided embeddings.
It demonstrates the entire workflow from data loading to evaluation.
"""

import json
from nl2cm.parse_args import parse_args
from nl2cm.evaluation import NL2CMEvaluator
from nl2cm.training import NL2CMTrainer
from nl2cm.model import NL2CMTranslator
from nl2cm.data_loader import load_nl2cm_data
from nl2cm.embed import get_embedding_dim
import os
import sys
import torch
from pathlib import Path


def test_data_loading():
    """Test data loading functionality."""
    print("=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    args = parse_args()

    # Load the dataframe
    data_path = os.path.join("datasets", "embeddings-dfs", args.dataset)
    # Test data loaders
    nl_cm_cols = [args.nl_col, args.cm_col]
    train_loader, val_loader, test_loader = load_nl2cm_data(
        data_path, nl_cm_cols, test_size=0.2, batch_size=args.batch_size, 
        num_workers=args.num_workers, limit=args.limit
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"NL batch shape: {batch['nlt'].shape}")
    print(f"CM batch shape: {batch['cmt'].shape}")

    print("✓ Data loading test passed\n")
    embedding_dim = get_embedding_dim(data_path)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def test_training(model, device, embedding_dim):
    """Test training functionality."""
    print("=" * 60)
    print("Testing Training")
    print("=" * 60)

    # Create trainer
    trainer = NL2CMTrainer(
        model=model,
        lr_generator=1e-4,
        lr_discriminator=4e-4,
        lambda_rec=15.0,
        lambda_cyc=15.0,
        lambda_vsp=2.0,
        lambda_adv=1.0,
        lambda_latent=1.0
    )

    # Create dummy data loaders
    from torch.utils.data import DataLoader, TensorDataset

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

    print("✓ Training test passed\n")
    return trainer


def test_evaluation(model, device, embedding_dim):
    """Test evaluation functionality."""
    print("=" * 60)
    print("Testing Evaluation")
    print("=" * 60)

    args = parse_args()

    # Create evaluator
    evaluator = NL2CMEvaluator(model, save_dir=args.save_dir)

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
    results = evaluator.evaluate_all(nlt_eval, cmt_eval)
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

    # Load real data
    args = parse_args()
    data_path = os.path.join("datasets", "embeddings-dfs", args.dataset)
    nl_cm_cols = [args.nl_col, args.cm_col]
    train_loader, val_loader, test_loader = load_nl2cm_data(
        data_path, nl_cm_cols, test_size=0.2, batch_size=args.batch_size, num_workers=args.num_workers, limit=args.limit)

    # Create model
    embedding_dim = get_embedding_dim(data_path)
    model = NL2CMTranslator(
        embedding_dim=embedding_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        adapter_depth=args.adapter_depth,
        backbone_depth=args.backbone_depth,
        dropout=args.dropout
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create trainer
    trainer = NL2CMTrainer(
        model=model,
        lr_generator=args.lr_generator,
        lr_discriminator=args.lr_discriminator,
        lambda_rec=args.lambda_rec,
        lambda_cyc=args.lambda_cyc,
        lambda_vsp=args.lambda_vsp,
        lambda_adv=args.lambda_adv,
        lambda_latent=args.lambda_latent,
        save_dir=args.save_dir
    )

    # Train for a few epochs
    print(f"Training for {args.epochs} epochs...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping_patience
    )

    # Evaluate on test data
    print("Evaluating on test data...")
    evaluator = NL2CMEvaluator(model, save_dir=args.save_dir)

    results = evaluator.evaluate_loader(test_loader)

    # Print results
    print("\n" + evaluator.create_evaluation_table(results))

    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'evaluation_results_{args.dataset}.json'), 'w') as f:
        json.dump(results, f)

    print("✓ Full pipeline test passed\n")


def main():
    """Run all tests."""
    print("Starting NL2CM Pipeline Tests")
    print("=" * 80)

    try:
        # Test 1: Data loading
        # embedding_dim = test_data_loading()

        # # Test 2: Model creation
        # model, device = test_model_creation(embedding_dim)

        # # Test 3: Training
        # trainer = test_training(model, device, embedding_dim)

        # # Test 4: Evaluation
        # evaluator = test_evaluation(model, device, embedding_dim)

        # Test 5: Full pipeline
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
