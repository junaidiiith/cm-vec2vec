"""
TensorBoard Demo for NL2CM

This script demonstrates how to use TensorBoard logging with the NL2CM package.
It runs a short training session and shows how to view the results in TensorBoard.
"""

from nl2cm.data_loader import load_nl2cm_data, create_evaluation_splits
from nl2cm import NL2CMTranslator, NL2CMTrainer, NL2CMEvaluator, create_tensorboard_logger
import os
import sys
import torch
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))


def main():
    """Run a demo training session with TensorBoard logging."""
    print("NL2CM TensorBoard Demo")
    print("=" * 50)

    # Load data
    data_path = "datasets/eamodelset_nl2cm_embeddings_df.pkl"
    train_loader, val_loader, test_loader = load_nl2cm_data(
        data_path, test_size=0.2)

    # Create model
    model = NL2CMTranslator(
        embedding_dim=1536,
        latent_dim=256,
        hidden_dim=512,
        adapter_depth=3,
        backbone_depth=4,
        dropout=0.1
    )

    device = 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'
    print(f"Using device: {device}")

    # Create TensorBoard logger
    tensorboard_logger = create_tensorboard_logger(
        log_dir='tensorboard_demo_logs',
        experiment_name='nl2cm_demo'
    )

    # Create trainer with TensorBoard logging
    trainer = NL2CMTrainer(
        model=model,
        device=device,
        lr_generator=1e-4,
        lr_discriminator=4e-4,
        lambda_rec=15.0,
        lambda_cyc=15.0,
        lambda_vsp=2.0,
        lambda_adv=1.0,
        lambda_latent=1.0,
        use_tensorboard=True,
        log_dir='tensorboard_demo_logs'
    )

    # Create evaluator with TensorBoard logging
    evaluator = NL2CMEvaluator(
        model=model,
        device=device,
        use_tensorboard=True,
        tensorboard_logger=trainer.tensorboard_logger
    )

    print("Starting training with TensorBoard logging...")
    print("This will log:")
    print("- Training losses (generator, discriminator, individual components)")
    print("- Validation losses")
    print("- Learning rates")
    print("- Model parameters and gradients")
    print("- Evaluation metrics")
    print("- Translation examples")

    # Train for a few epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        save_dir='demo_checkpoints',
        save_every=5
    )

    # Evaluate model
    print("\nEvaluating model...")
    nlt_eval, cmt_eval = create_evaluation_splits(data_path, 200)
    nlt_eval_tensor = torch.FloatTensor(nlt_eval).to(device)
    cmt_eval_tensor = torch.FloatTensor(cmt_eval).to(device)

    results = evaluator.evaluate_all(nlt_eval_tensor, cmt_eval_tensor)

    # Log hyperparameters and final metrics
    hparams = {
        'embedding_dim': 1536,
        'latent_dim': 256,
        'hidden_dim': 512,
        'lr_generator': 1e-4,
        'lr_discriminator': 4e-4,
        'lambda_rec': 15.0,
        'lambda_cyc': 15.0,
        'lambda_vsp': 2.0,
        'lambda_adv': 1.0,
        'lambda_latent': 1.0
    }

    tensorboard_logger.log_hyperparameters(hparams, results)

    # Close TensorBoard logger
    trainer.close_tensorboard()

    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)
    print(f"TensorBoard logs saved to: {tensorboard_logger.full_log_dir}")
    print("\nTo view the logs in TensorBoard, run:")
    print(f"tensorboard --logdir {tensorboard_logger.full_log_dir}")
    print("\nThen open your browser to: http://localhost:6006")
    print("\nYou'll see:")
    print("- Training/Validation loss curves")
    print("- Learning rate schedules")
    print("- Model parameter histograms")
    print("- Evaluation metrics")
    print("- Translation quality metrics")
    print("- Embedding visualizations")


if __name__ == '__main__':
    main()
