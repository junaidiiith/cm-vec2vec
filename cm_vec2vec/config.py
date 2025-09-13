"""
Configuration utilities for CMVec2Vec
"""

try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False
import json
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from TOML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix == '.toml':
        if not HAS_TOML:
            raise ImportError(
                "toml module is required to load TOML configuration files. Install with: pip install toml")
        with open(config_path, 'r') as f:
            config = toml.load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(
            f"Unsupported configuration file format: {config_path.suffix}")

    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to TOML or JSON file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.suffix == '.toml':
        if not HAS_TOML:
            raise ImportError(
                "toml module is required to save TOML configuration files. Install with: pip install toml")
        with open(config_path, 'w') as f:
            toml.dump(config, f)
    elif config_path.suffix == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(
            f"Unsupported configuration file format: {config_path.suffix}")


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for CMVec2Vec.

    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'embedding_dims': {
                'nl': 1536,
                'cm': 1536
            },
            'latent_dim': 256,
            'hidden_dim': 512,
            'adapter_depth': 3,
            'backbone_depth': 4,
            'dropout': 0.1,
            'use_conditioning': False,
            'cond_dim': 0,
            'normalize_embeddings': True,
            'weight_init': 'kaiming',
            'activation': 'silu'
        },
        'training': {
            'device': 'cuda',
            'lr_generator': 1e-4,
            'lr_discriminator': 4e-4,
            'loss_weights': {
                'reconstruction': 15.0,
                'cycle_consistency': 15.0,
                'vsp': 2.0,
                'adversarial': 1.0,
                'latent_adversarial': 1.0
            },
            'gan_type': 'least_squares',
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'use_scheduler': True,
            'warmup_steps': 1000,
            'epochs': 100,
            'batch_size': 32,
            'save_every': 10,
            'early_stopping_patience': 20
        },
        'data': {
            'data_path': 'data/embeddings.pkl',
            'domains': ['nl', 'cm'],
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            'normalize': True,
            'noise_level': 0.0
        },
        'evaluation': {
            'n_eval_samples': 1000,
            'k_values': [1, 5, 10],
            'n_clusters': None
        },
        'logging': {
            'use_wandb': False,
            'wandb_project': 'cm_vec2vec',
            'wandb_name': 'experiment',
            'log_every': 100,
            'save_dir': 'checkpoints'
        }
    }


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['model', 'training', 'data']

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: {section}")

    # Validate model configuration
    model_config = config['model']
    required_model_keys = ['embedding_dims', 'latent_dim', 'hidden_dim']

    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(
                f"Missing required model configuration key: {key}")

    # Validate training configuration
    training_config = config['training']
    required_training_keys = ['lr_generator', 'lr_discriminator', 'epochs']

    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(
                f"Missing required training configuration key: {key}")

    # Validate data configuration
    data_config = config['data']
    required_data_keys = ['data_path', 'domains']

    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Missing required data configuration key: {key}")

    return True


def create_config_from_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration from command line arguments.

    Args:
        args: Command line arguments dictionary

    Returns:
        Configuration dictionary
    """
    config = create_default_config()

    # Map common argument names to configuration keys
    arg_mapping = {
        'embedding_dims': 'model.embedding_dims',
        'latent_dim': 'model.latent_dim',
        'hidden_dim': 'model.hidden_dim',
        'adapter_depth': 'model.adapter_depth',
        'backbone_depth': 'model.backbone_depth',
        'dropout': 'model.dropout',
        'use_conditioning': 'model.use_conditioning',
        'cond_dim': 'model.cond_dim',
        'normalize_embeddings': 'model.normalize_embeddings',
        'weight_init': 'model.weight_init',
        'activation': 'model.activation',
        'device': 'training.device',
        'lr_generator': 'training.lr_generator',
        'lr_discriminator': 'training.lr_discriminator',
        'gan_type': 'training.gan_type',
        'weight_decay': 'training.weight_decay',
        'max_grad_norm': 'training.max_grad_norm',
        'use_scheduler': 'training.use_scheduler',
        'warmup_steps': 'training.warmup_steps',
        'epochs': 'training.epochs',
        'batch_size': 'training.batch_size',
        'save_every': 'training.save_every',
        'early_stopping_patience': 'training.early_stopping_patience',
        'data_path': 'data.data_path',
        'domains': 'data.domains',
        'test_size': 'data.test_size',
        'val_size': 'data.val_size',
        'random_state': 'data.random_state',
        'normalize': 'data.normalize',
        'noise_level': 'data.noise_level',
        'n_eval_samples': 'evaluation.n_eval_samples',
        'k_values': 'evaluation.k_values',
        'n_clusters': 'evaluation.n_clusters',
        'use_wandb': 'logging.use_wandb',
        'wandb_project': 'logging.wandb_project',
        'wandb_name': 'logging.wandb_name',
        'log_every': 'logging.log_every',
        'save_dir': 'logging.save_dir'
    }

    # Apply arguments to configuration
    for arg_name, config_path in arg_mapping.items():
        if arg_name in args:
            # Navigate to the correct location in the config
            keys = config_path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = args[arg_name]

    return config
