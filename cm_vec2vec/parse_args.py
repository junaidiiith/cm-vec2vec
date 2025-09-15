import argparse


def parse_args():
    """
    Parse command line arguments for CMVec2Vec training and evaluation.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train CMVec2Vec for NL2CM translation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # =============================================================================
    # CONFIGURATION AND DATA ARGUMENTS
    # =============================================================================
    config_group = parser.add_argument_group('Configuration and Data')
    config_group.add_argument(
        '--config',
        type=str,
        default='cm_vec2vec/configs/cm_vec2vec.toml',
        help='Path to configuration file'
    )
    config_group.add_argument(
        '--data_path',
        type=str,
        default='datasets/embeddings-dfs/',
        help='Path to NL2CM data file'
    )
    config_group.add_argument(
        '--dataset',
        type=str,
        default='bpmn',
        choices=['bpmn', 'archimate', 'ontouml'],
        help='Dataset name'
    )
    config_group.add_argument(
        '--nl_col',
        type=str,
        default='NL_Serialization_Emb_openai',
        help='Column name for NL embeddings'
    )
    config_group.add_argument(
        '--cm_col',
        type=str,
        default='CM_Serialization_Emb_openai',
        help='Column name for CM embeddings'
    )
    config_group.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing'
    )
    config_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    config_group.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )

    # =============================================================================
    # MODEL ARCHITECTURE ARGUMENTS
    # =============================================================================
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        '--embedding_dim',
        type=int,
        default=1536,
        help='Dimension of input embeddings'
    )
    model_group.add_argument(
        '--latent_dim',
        type=int,
        default=64,
        help='Dimension of shared latent space'
    )
    model_group.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='Hidden layer dimension for adapters and backbone'
    )
    model_group.add_argument(
        '--adapter_depth',
        type=int,
        default=3,
        help='Number of layers in adapter networks'
    )
    model_group.add_argument(
        '--backbone_depth',
        type=int,
        default=4,
        help='Number of layers in shared backbone network'
    )
    model_group.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate for regularization'
    )
    model_group.add_argument(
        '--weight_init',
        type=str,
        default='kaiming',
        choices=['kaiming', 'xavier', 'normal', 'uniform'],
        help='Weight initialization method'
    )
    model_group.add_argument(
        '--activation',
        type=str,
        default='silu',
        choices=['silu', 'relu', 'leaky_relu', 'gelu', 'tanh'],
        help='Activation function'
    )
    model_group.add_argument(
        '--use_conditioning',
        action='store_true',
        help='Enable conditioning on additional information'
    )
    model_group.add_argument(
        '--normalize_embeddings',
        action='store_true',
        help='Normalize embeddings to unit length'
    )

    # =============================================================================
    # TRAINING ARGUMENTS
    # =============================================================================
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    training_group.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Training batch size'
    )
    training_group.add_argument(
        '--lr_generator',
        type=float,
        default=1e-4,
        help='Learning rate for generator (translator)'
    )
    training_group.add_argument(
        '--lr_discriminator',
        type=float,
        default=4e-4,
        help='Learning rate for discriminators'
    )
    training_group.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay for L2 regularization'
    )
    training_group.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping'
    )
    training_group.add_argument(
        '--use_scheduler',
        action='store_true',
        help='Use learning rate scheduler'
    )
    training_group.add_argument(
        '--warmup_steps',
        type=int,
        default=1000,
        help='Number of warmup steps for scheduler'
    )

    # =============================================================================
    # LOSS WEIGHT ARGUMENTS
    # =============================================================================
    loss_group = parser.add_argument_group('Loss Function Weights')
    loss_group.add_argument(
        '--reconstruction_weight',
        type=float,
        default=10.0,
        help='Weight for reconstruction loss'
    )
    loss_group.add_argument(
        '--cycle_consistency_weight',
        type=float,
        default=10.0,
        help='Weight for cycle consistency loss'
    )
    loss_group.add_argument(
        '--vsp_weight',
        type=float,
        default=5.0,
        help='Weight for vector space preservation loss'
    )
    loss_group.add_argument(
        '--adversarial_weight',
        type=float,
        default=2.0,
        help='Weight for adversarial loss'
    )
    loss_group.add_argument(
        '--latent_adversarial_weight',
        type=float,
        default=2.0,
        help='Weight for latent adversarial loss'
    )
    loss_group.add_argument(
        '--correspondence_weight',
        type=float,
        default=20.0,
        help='Weight for correspondence loss'
    )
    loss_group.add_argument(
        '--cosine_correspondence_weight',
        type=float,
        default=5.0,
        help='Weight for cosine correspondence loss'
    )
    loss_group.add_argument(
        '--ranking_weight',
        type=float,
        default=15.0,
        help='Weight for ranking loss'
    )
    
    loss_group.add_argument(
        '--enhanced_losses',
        action='store_true',
        help='Use enhanced loss functions'
    )
    
    loss_group.add_argument(
        '--vsp_temperature',
        type=float,
        default=1.0,
        help='Temperature for vector space preservation loss'
    )
    loss_group.add_argument(
        '--focal_alpha',
        type=float,
        default=1.0,
        help='Alpha for focal adversarial loss'
    )
    loss_group.add_argument(
        '--focal_gamma',
        type=float,
        default=2.0,
        help='Gamma for focal adversarial loss'
    )
    loss_group.add_argument(
        '--cycle_margin',
        type=float,
        default=0.1,
        help='Margin for cycle consistency loss'
    )
    

    # =============================================================================
    # CHECKPOINTING AND SAVING ARGUMENTS
    # =============================================================================
    checkpoint_group = parser.add_argument_group('Checkpointing and Saving')
    checkpoint_group.add_argument(
        '--save_dir',
        type=str,
        default='logs/cm_vec2vec',
        help='Directory to save model checkpoints'
    )
    checkpoint_group.add_argument(
        '--save_every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    checkpoint_group.add_argument(
        '--early_stopping_patience',
        type=int,
        default=20,
        help='Early stopping patience (epochs)'
    )
    checkpoint_group.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to trained model checkpoint for evaluation'
    )

    # =============================================================================
    # EVALUATION AND OUTPUT ARGUMENTS
    # =============================================================================
    eval_group = parser.add_argument_group('Evaluation and Output')
    eval_group.add_argument(
        '--save_plots',
        action='store_true',
        help='Save embedding visualization plots'
    )
    eval_group.add_argument(
        '--save_table',
        action='store_true',
        help='Save evaluation table'
    )
    eval_group.add_argument(
        '--output_dir',
        type=str,
        default='logs/cm_vec2vec',
        help='Directory to save evaluation outputs'
    )

    return parser.parse_args()


def get_loss_weights(args):
    """
    Get loss weights from arguments, with individual weights taking precedence
    over the loss_weights dictionary.

    Args:
        args: Parsed arguments

    Returns:
        dict: Loss weights dictionary
    """
    # Start with individual weight arguments
    loss_weights = {
        'reconstruction': args.reconstruction_weight,
        'cycle_consistency': args.cycle_consistency_weight,
        'vsp': args.vsp_weight,
        'adversarial': args.adversarial_weight,
        'latent_adversarial': args.latent_adversarial_weight,
    }

    # Override with custom loss_weights if provided
    if args.loss_weights is not None:
        loss_weights.update(args.loss_weights)

    return loss_weights


def validate_args(args):
    """
    Validate argument values and raise errors for invalid combinations.

    Args:
        args: Parsed arguments

    Raises:
        ValueError: If arguments are invalid
    """
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not 0 <= args.test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")

    if args.lr_generator <= 0 or args.lr_discriminator <= 0:
        raise ValueError("Learning rates must be positive")

    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("dropout must be between 0 and 1")

    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    # Validate loss weights are non-negative
    loss_weights = get_loss_weights(args)
    for name, weight in loss_weights.items():
        if weight < 0:
            raise ValueError(
                f"{name}_weight must be non-negative, got {weight}")


if __name__ == "__main__":
    # Example usage and validation
    args = parse_args()
    validate_args(args)
    print("Arguments parsed successfully!")
    print(f"Loss weights: {get_loss_weights(args)}")
