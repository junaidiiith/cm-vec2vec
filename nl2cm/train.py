"""
NL2CM Training Script

Main training script for the NL2CM translation model based on the vec2vec approach.
This script handles data loading, model training, and evaluation.
"""

import argparse
import os
import json
import torch

from data_loader import load_nl2cm_data, create_evaluation_splits
from model import NL2CMTranslator
from training import NL2CMTrainer
from evaluation import NL2CMEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train NL2CM translation model')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                        default='datasets/eamodelset_nl2cm_embeddings_df.pkl',
                        help='Path to the pickle file containing embeddings')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')

    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=1536,
                        help='Embedding dimension')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent space dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--adapter_depth', type=int, default=3,
                        help='Adapter network depth')
    parser.add_argument('--backbone_depth', type=int, default=4,
                        help='Backbone network depth')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--use_conditioning', action='store_true',
                        help='Use conditioning in the backbone')
    parser.add_argument('--cond_dim', type=int, default=0,
                        help='Conditioning dimension')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr_generator', type=float, default=1e-4,
                        help='Learning rate for generator')
    parser.add_argument('--lr_discriminator', type=float, default=4e-4,
                        help='Learning rate for discriminator')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # Loss weights
    parser.add_argument('--lambda_rec', type=float, default=15.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--lambda_cyc', type=float, default=15.0,
                        help='Cycle consistency loss weight')
    parser.add_argument('--lambda_vsp', type=float, default=2.0,
                        help='Vector space preservation loss weight')
    parser.add_argument('--lambda_adv', type=float, default=1.0,
                        help='Adversarial loss weight')
    parser.add_argument('--lambda_latent', type=float, default=1.0,
                        help='Latent adversarial loss weight')

    # Training options
    parser.add_argument('--save_dir', type=str, default='checkpoints/nl2cm',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')

    # TensorBoard options
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use TensorBoard logging')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard_logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment (default: auto-generated)')

    # Evaluation arguments
    parser.add_argument('--eval_samples', type=int, default=1000,
                        help='Number of samples for evaluation')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
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
    train_loader, val_loader, test_loader = load_nl2cm_data(
        args.data_path, args.test_size, args.random_state
    )

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

    # Create evaluation data
    print("Creating evaluation data...")
    nlt_eval, cmt_eval = create_evaluation_splits(
        args.data_path, args.eval_samples)
    nlt_eval_tensor = torch.FloatTensor(nlt_eval).to(device)
    cmt_eval_tensor = torch.FloatTensor(cmt_eval).to(device)

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
    results = evaluator.evaluate_all(nlt_eval_tensor, cmt_eval_tensor)

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


if __name__ == '__main__':
    main()
