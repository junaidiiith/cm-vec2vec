"""
NL2CM Training Script

Main training script for the NL2CM translation model based on the vec2vec approach.
This script handles data loading, model training, and evaluation.
"""

import os
import json
import torch

from nl2cm.data_loader import load_nl2cm_data, create_evaluation_splits
from nl2cm.model import NL2CMTranslator
from nl2cm.utils import get_device
from nl2cm.training import NL2CMTrainer
from nl2cm.evaluation import NL2CMEvaluator
from nl2cm.parse_args import parse_args


def main():
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
        args.test_size, args.random_state
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


if __name__ == '__main__':
    main()
