"""
Evaluation example for CMVec2Vec
"""

import os
import torch
from pathlib import Path

from cm_vec2vec import CMVec2VecTranslator, CMVec2VecEvaluator
from cm_vec2vec.data_loader import load_nl2cm_data
from cm_vec2vec.utils import get_device
from cm_vec2vec.parse_args import parse_args


def main():
    args = parse_args()

    print("Evaluation Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Data path: {args.data_path}")
    print(f"  NL column: {args.nl_col}")
    print(f"  CM column: {args.cm_col}")
    print(f"  Device: {get_device()}")
    print()


    # Load model
    print("Loading trained model...")
    checkpoint = torch.load(args.model_path, map_location=get_device())

    # Create model with same architecture
    model = CMVec2VecTranslator(
        embedding_dim=checkpoint.get('embedding_dim', 1536),
        latent_dim=checkpoint.get('latent_dim', 256),
        hidden_dim=checkpoint.get('hidden_dim', 512),
        adapter_depth=checkpoint.get('adapter_depth', 3),
        backbone_depth=checkpoint.get('backbone_depth', 4),
        dropout=checkpoint.get('dropout', 0.1),
        use_conditioning=checkpoint.get('use_conditioning', False),
        normalize_embeddings=checkpoint.get('normalize_embeddings', True),
        weight_init=checkpoint.get('weight_init', 'kaiming'),
        activation=checkpoint.get('activation', 'silu')
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(get_device())
    model.eval()

    print("Model loaded successfully!")
    print()
    data_path = os.path.join(args.data_path, args.dataset_name)
    nl_cm_cols = [args.nl_col, args.cm_col]
    data = load_nl2cm_data(
        data_path, 
        nl_cm_cols, 
        args.test_size, 
        args.seed, 
        args.batch_size, 
        args.num_workers
    )
    test_loader = data['test_loader']
    # Create evaluator
    evaluator = CMVec2VecEvaluator(model)

    # Evaluate NL to CM translation
    nlt_to_cmt_results = evaluator.evaluate_batch(test_loader)

    print(evaluator.create_evaluation_table(nlt_to_cmt_results))

    # Save results
    results_save_path = Path('evaluation_results.json')
    import json
    with open(results_save_path, 'w') as f:
        json.dump(nlt_to_cmt_results, f, indent=2)

    print(f"\nResults saved to: {results_save_path}")

    # Create embedding visualizations if requested
    if args.save_plots:
        print("\nCreating embedding visualizations...")

        # Create plots directory
        plots_dir = Path('evaluation_plots')
        plots_dir.mkdir(exist_ok=True)

        # Plot original embeddings
        nlt_embeddings = torch.cat([batch['nlt'] for batch in test_loader], dim=0)
        cmt_embeddings = torch.cat([batch['cmt'] for batch in test_loader], dim=0)
        nlt_embeddings_translated = evaluator.get_nlt_to_cmt_embeddings(test_loader)
        cmt_embeddings_translated = evaluator.get_cmt_to_nlt_embeddings(test_loader)
        
        for method in ['tsne', 'pca', 'umap']:
            evaluator.plot_embeddings(
                nlt_embeddings=nlt_embeddings,
                cmt_embeddings=cmt_embeddings,
                translated_nlt=nlt_embeddings_translated,
                translated_cmt=cmt_embeddings_translated,
                method=method,
                save_path=plots_dir / f'original_embeddings_{method}.png'
            )
        
        print(f"Plots saved to: {plots_dir}")


if __name__ == '__main__':
    main()
