"""
Test NL2CM Pipeline

This script tests the complete NL2CM pipeline on the provided embeddings.
It demonstrates the entire workflow from data loading to evaluation.
"""

import json
import sys
import torch
import os
from torch.utils.data import DataLoader, TensorDataset

from nl2cm.utils import get_device
from nl2cm.parse_args import parse_args
from nl2cm.evaluation import NL2CMEvaluator
from nl2cm.training import NL2CMTrainer
from nl2cm.model import NL2CMTranslator
from nl2cm.data_loader import create_evaluation_splits, load_nl2cm_data



def load_model(checkpoint_path: str) -> NL2CMTranslator:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=get_device())

    # Extract model configuration from checkpoint
    model_state = checkpoint['model_state_dict']

    # Infer model configuration from state dict
    embedding_dim = None
    latent_dim = None

    for key, value in model_state.items():
        if 'nlt_adapter.network.0.weight' in key:
            embedding_dim = value.shape[1]
        elif 'nlt_adapter.network.0.weight' in key:
            latent_dim = value.shape[0]

    # Default values if not found
    if embedding_dim is None:
        embedding_dim = 1536
    if latent_dim is None:
        latent_dim = 256

    # Create model
    model = NL2CMTranslator(
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        hidden_dim=512,
        adapter_depth=3,
        backbone_depth=4,
        dropout=0.1
    )

    # Load state dict
    model.load_state_dict(model_state)
    model.to(get_device())
    model.eval()

    return model


def create_baseline_comparison(nlt_emb: torch.Tensor, cmt_emb: torch.Tensor) -> dict:
    """Create baseline comparison results."""
    # Identity baseline
    identity_cosine = torch.nn.functional.cosine_similarity(
        nlt_emb, cmt_emb, dim=1).mean().item()

    # Random baseline (mean rank should be around N/2)
    n_samples = len(nlt_emb)
    random_mean_rank = n_samples / 2

    return {
        'identity_cosine_similarity': identity_cosine,
        'random_mean_rank': random_mean_rank
    }


def create_evaluation_table(results: dict, baseline_results: dict) -> str:
    """Create a comprehensive evaluation table."""
    table = "=" * 100 + "\n"
    table += "NL2CM Translation Evaluation Results\n"
    table += "=" * 100 + "\n\n"

    # Basic translation metrics
    table += "Translation Quality Metrics:\n"
    table += "-" * 50 + "\n"
    table += f"{'Metric':<30} {'Value':<15} {'Baseline':<15} {'Improvement':<15}\n"
    table += "-" * 50 + "\n"

    cosine_sim = results.get('cosine_similarity', 0.0)
    baseline_cosine = baseline_results.get('identity_cosine_similarity', 0.0)
    cosine_improvement = cosine_sim - baseline_cosine

    table += f"{'Cosine Similarity':<30} {cosine_sim:<15.4f} {baseline_cosine:<15.4f} {cosine_improvement:<15.4f}\n"

    mean_rank = results.get('mean_rank', float('inf'))
    baseline_rank = baseline_results.get('random_mean_rank', float('inf'))
    rank_improvement = baseline_rank - \
        mean_rank if mean_rank != float('inf') else 0

    table += f"{'Mean Rank':<30} {mean_rank:<15.2f} {baseline_rank:<15.2f} {rank_improvement:<15.2f}\n"

    top1 = results.get('top_1_accuracy', 0.0)
    table += f"{'Top-1 Accuracy':<30} {top1:<15.4f} {'0.0000':<15} {top1:<15.4f}\n"

    top5 = results.get('top_5_accuracy', 0.0)
    table += f"{'Top-5 Accuracy':<30} {top5:<15.4f} {'0.0000':<15} {top5:<15.4f}\n"

    mrr = results.get('mrr', 0.0)
    table += f"{'MRR':<30} {mrr:<15.4f} {'0.0000':<15} {mrr:<15.4f}\n\n"

    # Cycle consistency
    table += "Cycle Consistency:\n"
    table += "-" * 50 + "\n"
    table += f"{'NL Cycle Similarity':<30} {results.get('nlt_cycle_similarity', 0.0):<15.4f}\n"
    table += f"{'CM Cycle Similarity':<30} {results.get('cmt_cycle_similarity', 0.0):<15.4f}\n"
    table += f"{'Mean Cycle Similarity':<30} {results.get('mean_cycle_similarity', 0.0):<15.4f}\n\n"

    # Geometry preservation
    table += "Geometry Preservation:\n"
    table += "-" * 50 + "\n"
    table += f"{'NL Geometry Correlation':<30} {results.get('nlt_geometry_correlation', 0.0):<15.4f}\n"
    table += f"{'CM Geometry Correlation':<30} {results.get('cmt_geometry_correlation', 0.0):<15.4f}\n"
    table += f"{'Mean Geometry Correlation':<30} {results.get('mean_geometry_correlation', 0.0):<15.4f}\n\n"

    # Classification metrics (if available)
    if 'translated_ari' in results:
        table += "Classification Performance:\n"
        table += "-" * 50 + "\n"
        table += f"{'Translated ARI':<30} {results.get('translated_ari', 0.0):<15.4f}\n"
        table += f"{'Translated NMI':<30} {results.get('translated_nmi', 0.0):<15.4f}\n"
        table += f"{'Original ARI':<30} {results.get('original_ari', 0.0):<15.4f}\n"
        table += f"{'Original NMI':<30} {results.get('original_nmi', 0.0):<15.4f}\n"
        table += f"{'ARI Improvement':<30} {results.get('ari_improvement', 0.0):<15.4f}\n"
        table += f"{'NMI Improvement':<30} {results.get('nmi_improvement', 0.0):<15.4f}\n\n"

    table += "=" * 100 + "\n"

    return table


def train_model():
    """Main training function."""
    args = parse_args()
    data_path = os.path.join(args.data_path, args.dataset)
    nl_cm_cols = [args.nl_col, args.cm_col]
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_path}")

    # Load data
    print("Loading data...")
    data = load_nl2cm_data(
        data_path, nl_cm_cols, 
        args.test_size, args.random_state,
        limit=args.limit, batch_size=args.batch_size, num_workers=args.num_workers
    )
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("Creating model...")
    model = NL2CMTranslator(
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        adapter_depth=args.adapter_depth,
        backbone_depth=args.backbone_depth,
        dropout=args.dropout,
        use_conditioning=args.use_conditioning,
        cond_dim=args.cond_dim
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer with TensorBoard logging
    trainer = NL2CMTrainer(
        model=model,
        device=device,
        lr_generator=args.lr_generator,
        lr_discriminator=args.lr_discriminator,
        lambda_rec=args.lambda_rec,
        lambda_cyc=args.lambda_cyc,
        lambda_vsp=args.lambda_vsp,
        lambda_adv=args.lambda_adv,
        lambda_latent=args.lambda_latent,
        weight_decay=args.weight_decay,
        use_tensorboard=args.use_tensorboard,
        log_dir=args.tensorboard_dir
    )

    # Create evaluator with TensorBoard logging
    evaluator = NL2CMEvaluator(
        model=model,
        device=device,
        use_tensorboard=args.use_tensorboard,
        tensorboard_logger=trainer.tensorboard_logger
    )


    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping_patience
    )

    # Final evaluation
    print("Performing final evaluation...")
    # Create evaluation data
    print("Creating evaluation data...")
    nlt_eval, cmt_eval = create_evaluation_splits(data_path, nl_cm_cols, args.eval_samples)
    nlt_eval_tensor = torch.FloatTensor(nlt_eval).to(device)
    cmt_eval_tensor = torch.FloatTensor(cmt_eval).to(device)
    results = evaluator.evaluate(nlt_eval_tensor, cmt_eval_tensor)

    # Print results
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    print(evaluator.create_evaluation_table(results))

    # Save results
    results_path = os.path.join(args.save_dir, 'final_results.json')
    evaluator.save_results(results, results_path)

    # Plot training curves
    plot_path = os.path.join(args.save_dir, 'training_curves.png')
    evaluator.plot_training_curves(
        history['train_losses'], history['val_losses'], plot_path)

    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Close TensorBoard logger
    trainer.close_tensorboard()

    print(f"Training completed. Results saved to {args.save_dir}")
    if args.use_tensorboard:
        print(f"TensorBoard logs saved to {args.tensorboard_dir}")
        print(
            f"To view logs, run: tensorboard --logdir {args.tensorboard_dir}")



def evaluate_model():
    """Main evaluation function."""
    args = parse_args()

    data_path = os.path.join(args.data_path, args.dataset)
    nl_cm_cols = [args.nl_col, args.cm_col]
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint_path)
    print("Model loaded successfully")

    # Create evaluator
    evaluator = NL2CMEvaluator(model)

    # Load evaluation data
    print("Loading evaluation data...")
    nlt_eval, cmt_eval = create_evaluation_splits(data_path, nl_cm_cols, args.eval_samples)
    nlt_eval_tensor = torch.FloatTensor(nlt_eval).to(device)
    cmt_eval_tensor = torch.FloatTensor(cmt_eval).to(device)

    print(f"Evaluation data: {len(nlt_eval)} samples")

    # Compute baseline results
    print("Computing baseline results...")
    baseline_results = create_baseline_comparison(nlt_eval_tensor, cmt_eval_tensor)

    # Evaluate model
    print("Evaluating model...")
    results = evaluator.evaluate(nlt_eval_tensor, cmt_eval_tensor)

    # Create evaluation table
    table = create_evaluation_table(results, baseline_results)
    print(table)

    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    evaluator.save_results(results, results_path)

    # Save baseline results
    baseline_path = os.path.join(args.output_dir, 'baseline_results.json')
    with open(baseline_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)

    # Save evaluation table
    table_path = os.path.join(args.output_dir, 'evaluation_table.txt')
    with open(table_path, 'w') as f:
        f.write(table)

    print(f"Evaluation completed. Results saved to {args.output_dir}")


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


def test_end_to_end():
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
        test_end_to_end()
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
