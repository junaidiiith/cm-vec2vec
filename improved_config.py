"""
Improved configuration recommendations for CMVec2Vec training
"""

# Improved loss weights based on evaluation results
IMPROVED_LOSS_WEIGHTS = {
    # Increase reconstruction weight - critical for embedding quality
    'reconstruction': 25.0,  # Increased from 15.0

    # Increase cycle consistency - important for bidirectional translation
    'cycle_consistency': 20.0,  # Increased from 15.0

    # Increase VSP weight - helps with geometry preservation
    'vsp': 5.0,  # Increased from 2.0

    # Moderate adversarial weight - balance between realism and stability
    'adversarial': 2.0,  # Increased from 1.0

    # Keep latent adversarial moderate
    'latent_adversarial': 1.5,  # Slightly increased from 1.0
}

# Alternative configurations for experimentation
EXPERIMENTAL_CONFIGS = {
    'conservative': {
        'reconstruction': 20.0,
        'cycle_consistency': 18.0,
        'vsp': 4.0,
        'adversarial': 1.5,
        'latent_adversarial': 1.0,
    },

    'aggressive': {
        'reconstruction': 30.0,
        'cycle_consistency': 25.0,
        'vsp': 8.0,
        'adversarial': 3.0,
        'latent_adversarial': 2.0,
    },

    'balanced': {
        'reconstruction': 22.0,
        'cycle_consistency': 20.0,
        'vsp': 6.0,
        'adversarial': 2.5,
        'latent_adversarial': 1.8,
    }
}

# Training hyperparameters for better performance
TRAINING_IMPROVEMENTS = {
    'learning_rates': {
        'generator_lr': 5e-5,  # Reduced from default for stability
        'discriminator_lr': 1e-4,  # Reduced from default
    },

    'optimization': {
        'weight_decay': 0.001,  # Reduced for less regularization
        'max_grad_norm': 0.5,  # Reduced for gentler clipping
        'warmup_steps': 2000,  # Increased warmup
    },

    'scheduling': {
        'use_cosine_annealing': True,
        'cosine_eta_min': 1e-6,
        'patience': 15,  # Reduced early stopping patience
    }
}

# Model architecture improvements
ARCHITECTURE_IMPROVEMENTS = {
    'embedding_dim': 512,  # Consider increasing if current < 512
    'hidden_dim': 1024,  # Increase hidden dimension
    'num_layers': 4,  # Add more layers
    'dropout': 0.1,  # Add dropout for regularization
    'use_layer_norm': True,  # Add layer normalization
    'use_residual_connections': True,  # Add skip connections
}

# Data augmentation strategies
DATA_IMPROVEMENTS = {
    'noise_level': 0.02,  # Slightly increase noise
    'mixup_alpha': 0.2,  # Add mixup augmentation
    'embedding_normalization': True,  # Normalize embeddings
    'batch_size': 64,  # Smaller batch size for better gradients
}
