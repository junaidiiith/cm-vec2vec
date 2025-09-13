import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CMVec2Vec for NL2CM translation')
    parser.add_argument('--config', type=str, default='cm_vec2vec/configs/cm_vec2vec.toml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str,
                        default='datasets/embeddings-dfs/',
                        help='Path to NL2CM data file')
    parser.add_argument('--nl_col', type=str, default='NL_Serialization_Emb_openai',
                        help='Column name for NL embeddings')
    parser.add_argument('--cm_col', type=str, default='CM_Serialization_Emb_openai',
                        help='Column name for CM embeddings')
    parser.add_argument('--dataset_name', type=str, default='bpmn',
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--save_dir', type=str, default='logs/cm_vec2vec',
                        help='Directory to save checkpoints')
    parser.add_argument('--embedding_dim', type=int, required=False, default=1536,
                        help='Embedding dimension')
    parser.add_argument('--latent_dim', type=int, required=False, default=256,
                        help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, required=False, default=512,
                        help='Hidden dimension')
    parser.add_argument('--adapter_depth', type=int, required=False, default=3,
                        help='Adapter depth')
    parser.add_argument('--backbone_depth', type=int, required=False, default=4,
                        help='Backbone depth')
    parser.add_argument('--dropout', type=float, required=False, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--use_conditioning', action='store_true',
                        help='Use conditioning')
    parser.add_argument('--normalize_embeddings', action='store_true',
                        help='Normalize embeddings')
    parser.add_argument('--weight_init', type=str, required=False, default='kaiming',
                        help='Weight initialization')
    parser.add_argument('--activation', type=str, required=False, default='silu',
                        help='Activation function')
    parser.add_argument('--lr_generator', type=float, required=False, default=1e-4,
                        help='Learning rate for generator')
    parser.add_argument('--lr_discriminator', type=float, required=False, default=4e-4,
                        help='Learning rate for discriminator')
    parser.add_argument('--loss_weights', type=dict, required=False, default=None,
                        help='Loss weights')
    parser.add_argument('--weight_decay', type=float, required=False, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, required=False, default=1.0,
                        help='Maximum gradient norm')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use scheduler')
    parser.add_argument('--warmup_steps', type=int, required=False, default=1000,
                        help='Warmup steps')
    parser.add_argument('--early_stopping_patience', type=int, required=False, default=20,
                        help='Early stopping patience')
    parser.add_argument('--save_every', type=int, required=False, default=10,
                        help='Save every')
    parser.add_argument('--test_size', type=float, required=False, default=0.2,
                        help='Test size')
    parser.add_argument('--seed', type=int, required=False, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, required=False, default=4,
                        help='Number of workers')
    
    parser.add_argument('--reconstruction_weight', type=float, required=False, default=15.0,
                        help='Reconstruction weight')
    parser.add_argument('--cycle_consistency_weight', type=float, required=False, default=15.0,
                        help='Cycle consistency weight')
    parser.add_argument('--vsp_weight', type=float, required=False, default=2.0,
                        help='VSP weight')
    parser.add_argument('--adversarial_weight', type=float, required=False, default=2.0,
                        help='Adversarial weight')
    parser.add_argument('--latent_adversarial_weight', type=float, required=False, default=5.0,
                        help='Latent adversarial weight')
    
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save embedding visualization plots')
    parser.add_argument('--output_path', type=str, required=False, default='logs/cm_vec2vec',
                        help='Path to save output')
    
    return parser.parse_args()