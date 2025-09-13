"""
Basic training example for CMVec2Vec
"""

import argparse
from pathlib import Path

from cm_vec2vec import CMVec2VecTranslator, CMVec2VecTrainer
from cm_vec2vec.data_loader import load_nl2cm_data
from cm_vec2vec.config import load_config


def create_default_config():
    """Create default configuration."""
    return {
        'model': {
            'embedding_dim': 1536,
            'latent_dim': 256,
            'hidden_dim': 512,
            'adapter_depth': 3,
            'backbone_depth': 4,
            'dropout': 0.1,
            'use_conditioning': False,
            'normalize_embeddings': True,
            'weight_init': 'kaiming',
            'activation': 'silu'
        },
        'training': {
            'lr_generator': 1e-4,
            'lr_discriminator': 4e-4,
            'epochs': 100,
            'batch_size': 32,
            'device': 'cuda',
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'use_scheduler': True,
            'warmup_steps': 1000,
            'early_stopping_patience': 20
        },
        'loss_weights': {
            'reconstruction': 15.0,
            'cycle_consistency': 15.0,
            'vsp': 2.0,
            'adversarial': 1.0,
            'latent_adversarial': 1.0
        },
        'data': {
            'data_path': '',
            'test_size': 0.2,
            'random_state': 42,
            'normalize': True,
            'noise_level': 0.01
        },
        'logging': {
            'save_dir': 'checkpoints',
            'save_every': 10
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train CMVec2Vec model')
    parser.add_argument('--config', type=str, default='configs/default.toml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr_generator', type=float, default=1e-4,
                        help='Generator learning rate')
    parser.add_argument('--lr_discriminator', type=float, default=4e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')

    args = parser.parse_args()

    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        config = create_default_config()

    # Override with command line arguments
    config['data']['data_path'] = args.data_path
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['lr_generator'] = args.lr_generator
    config['training']['lr_discriminator'] = args.lr_discriminator
    config['training']['device'] = args.device
    config['logging']['save_dir'] = args.save_dir

    print("Configuration:")
    print(f"  Data path: {config['data']['data_path']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Device: {config['training']['device']}")
    print()

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = load_nl2cm_data(
        data_path=config['data']['data_path'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print()

    # Create model
    print("Creating model...")
    model = CMVec2VecTranslator(
        embedding_dim=config['model']['embedding_dim'],
        latent_dim=config['model']['latent_dim'],
        hidden_dim=config['model']['hidden_dim'],
        adapter_depth=config['model']['adapter_depth'],
        backbone_depth=config['model']['backbone_depth'],
        dropout=config['model']['dropout'],
        use_conditioning=config['model']['use_conditioning'],
        normalize_embeddings=config['model']['normalize_embeddings'],
        weight_init=config['model']['weight_init'],
        activation=config['model']['activation']
    )

    print(
        f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create trainer
    print("Creating trainer...")
    trainer = CMVec2VecTrainer(
        model=model,
        device=config['training']['device'],
        lr_generator=config['training']['lr_generator'],
        lr_discriminator=config['training']['lr_discriminator'],
        loss_weights=config['loss_weights'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        use_scheduler=config['training']['use_scheduler'],
        warmup_steps=config['training']['warmup_steps']
    )

    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        save_dir=config['logging']['save_dir'],
        save_every=config['logging']['save_every'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )

    print("Training completed!")
    print(f"Final training loss: {history['train_losses'][-1]['total']:.4f}")
    if history['val_losses']:
        print(
            f"Final validation loss: {history['val_losses'][-1]['total']:.4f}")


if __name__ == '__main__':
    main()
