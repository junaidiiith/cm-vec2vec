"""
NL2CM Evaluation Script

This script evaluates a trained NL2CM model and generates comprehensive results
matching the vec2vec paper evaluation format.
"""

import os
import json
import torch

from nl2cm.data_loader import create_evaluation_splits
from nl2cm.model import NL2CMTranslator
from nl2cm.evaluation import NL2CMEvaluator
from nl2cm.utils import get_device
from nl2cm.parse_args import parse_args


def load_model(checkpoint_path: str) -> NL2CMTranslator:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=get_device())

    # Extract model configuration from checkpoint
    model_state = checkpoint['model_state_dict']

    # Infer model configuration from state dict
    embedding_dim = None
    latent_dim = None

    for key, value in model_state.items():
        if 'nlt_adapter.network.0.weight' in key:
            embedding_dim = value.shape[1]
        elif 'nlt_adapter.network.0.weight' in key:
            latent_dim = value.shape[0]

    # Default values if not found
    if embedding_dim is None:
        embedding_dim = 1536
    if latent_dim is None:
        latent_dim = 256

    # Create model
    model = NL2CMTranslator(
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        hidden_dim=512,
        adapter_depth=3,
        backbone_depth=4,
        dropout=0.1
    )

    # Load state dict
    model.load_state_dict(model_state)
    model.to(get_device())
    model.eval()

    return model


def create_baseline_comparison(nlt_emb: torch.Tensor, cmt_emb: torch.Tensor) -> dict:
    """Create baseline comparison results."""
    # Identity baseline
    identity_cosine = torch.nn.functional.cosine_similarity(
        nlt_emb, cmt_emb, dim=1).mean().item()

    # Random baseline (mean rank should be around N/2)
    n_samples = len(nlt_emb)
    random_mean_rank = n_samples / 2

    return {
        'identity_cosine_similarity': identity_cosine,
        'random_mean_rank': random_mean_rank
    }


def create_evaluation_table(results: dict, baseline_results: dict) -> str:
    """Create a comprehensive evaluation table."""
    table = "=" * 100 + "\n"
    table += "NL2CM Translation Evaluation Results\n"
    table += "=" * 100 + "\n\n"

    # Basic translation metrics
    table += "Translation Quality Metrics:\n"
    table += "-" * 50 + "\n"
    table += f"{'Metric':<30} {'Value':<15} {'Baseline':<15} {'Improvement':<15}\n"
    table += "-" * 50 + "\n"

    cosine_sim = results.get('cosine_similarity', 0.0)
    baseline_cosine = baseline_results.get('identity_cosine_similarity', 0.0)
    cosine_improvement = cosine_sim - baseline_cosine

    table += f"{'Cosine Similarity':<30} {cosine_sim:<15.4f} {baseline_cosine:<15.4f} {cosine_improvement:<15.4f}\n"

    mean_rank = results.get('mean_rank', float('inf'))
    baseline_rank = baseline_results.get('random_mean_rank', float('inf'))
    rank_improvement = baseline_rank - \
        mean_rank if mean_rank != float('inf') else 0

    table += f"{'Mean Rank':<30} {mean_rank:<15.2f} {baseline_rank:<15.2f} {rank_improvement:<15.2f}\n"

    top1 = results.get('top_1_accuracy', 0.0)
    table += f"{'Top-1 Accuracy':<30} {top1:<15.4f} {'0.0000':<15} {top1:<15.4f}\n"

    top5 = results.get('top_5_accuracy', 0.0)
    table += f"{'Top-5 Accuracy':<30} {top5:<15.4f} {'0.0000':<15} {top5:<15.4f}\n"

    mrr = results.get('mrr', 0.0)
    table += f"{'MRR':<30} {mrr:<15.4f} {'0.0000':<15} {mrr:<15.4f}\n\n"

    # Cycle consistency
    table += "Cycle Consistency:\n"
    table += "-" * 50 + "\n"
    table += f"{'NL Cycle Similarity':<30} {results.get('nlt_cycle_similarity', 0.0):<15.4f}\n"
    table += f"{'CM Cycle Similarity':<30} {results.get('cmt_cycle_similarity', 0.0):<15.4f}\n"
    table += f"{'Mean Cycle Similarity':<30} {results.get('mean_cycle_similarity', 0.0):<15.4f}\n\n"

    # Geometry preservation
    table += "Geometry Preservation:\n"
    table += "-" * 50 + "\n"
    table += f"{'NL Geometry Correlation':<30} {results.get('nlt_geometry_correlation', 0.0):<15.4f}\n"
    table += f"{'CM Geometry Correlation':<30} {results.get('cmt_geometry_correlation', 0.0):<15.4f}\n"
    table += f"{'Mean Geometry Correlation':<30} {results.get('mean_geometry_correlation', 0.0):<15.4f}\n\n"

    # Classification metrics (if available)
    if 'translated_ari' in results:
        table += "Classification Performance:\n"
        table += "-" * 50 + "\n"
        table += f"{'Translated ARI':<30} {results.get('translated_ari', 0.0):<15.4f}\n"
        table += f"{'Translated NMI':<30} {results.get('translated_nmi', 0.0):<15.4f}\n"
        table += f"{'Original ARI':<30} {results.get('original_ari', 0.0):<15.4f}\n"
        table += f"{'Original NMI':<30} {results.get('original_nmi', 0.0):<15.4f}\n"
        table += f"{'ARI Improvement':<30} {results.get('ari_improvement', 0.0):<15.4f}\n"
        table += f"{'NMI Improvement':<30} {results.get('nmi_improvement', 0.0):<15.4f}\n\n"

    table += "=" * 100 + "\n"

    return table


def main():
    """Main evaluation function."""
    args = parse_args()

    data_path = os.path.join(args.data_path, args.dataset)
    nl_cm_cols = [args.nl_col, args.cm_col]
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint_path)
    print("Model loaded successfully")

    # Create evaluator
    evaluator = NL2CMEvaluator(model)

    # Load evaluation data
    print("Loading evaluation data...")
    nlt_eval, cmt_eval = create_evaluation_splits(data_path, nl_cm_cols, args.eval_samples)
    nlt_eval_tensor = torch.FloatTensor(nlt_eval).to(device)
    cmt_eval_tensor = torch.FloatTensor(cmt_eval).to(device)

    print(f"Evaluation data: {len(nlt_eval)} samples")

    # Compute baseline results
    print("Computing baseline results...")
    baseline_results = create_baseline_comparison(nlt_eval_tensor, cmt_eval_tensor)

    # Evaluate model
    print("Evaluating model...")
    results = evaluator.evaluate(nlt_eval_tensor, cmt_eval_tensor)

    # Create evaluation table
    table = create_evaluation_table(results, baseline_results)
    print(table)

    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    evaluator.save_results(results, results_path)

    # Save baseline results
    baseline_path = os.path.join(args.output_dir, 'baseline_results.json')
    with open(baseline_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)

    # Save evaluation table
    table_path = os.path.join(args.output_dir, 'evaluation_table.txt')
    with open(table_path, 'w') as f:
        f.write(table)

    print(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
