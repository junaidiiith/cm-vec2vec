import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train NL2CM translation model')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                        default='datasets/embeddings-dfs',
                        help='Path to the pickle file containing embeddings')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument("--dataset", type=str, default="archimate")
    parser.add_argument("--nl_col", type=str, default="NL_Serialization_Emb")
    parser.add_argument("--cm_col", type=str, default="CM_Serialization_Emb")

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
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save model every N epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience')

    # TensorBoard options
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use TensorBoard logging')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard_logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment (default: auto-generated)')

    # Evaluation arguments
    parser.add_argument('--eval_samples', type=int, default=20000,
                        help='Number of samples for evaluation')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')

    return parser.parse_args()
