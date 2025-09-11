"""
Generate Evaluation Tables

This script generates comprehensive evaluation tables in the format of the vec2vec paper,
comparing NL2CM translation against baselines.
"""

from nl2cm.evaluation import NL2CMEvaluator
from nl2cm.model import NL2CMTranslator
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))


def load_best_model(checkpoint_dir: str, device: str = 'cuda') -> NL2CMTranslator:
    """Load the best trained model."""
    best_path = os.path.join(checkpoint_dir, 'nl2cm_best.pt')
    if not os.path.exists(best_path):
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir(
            checkpoint_dir) if f.startswith('nl2cm_epoch_')]
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        latest_checkpoint = sorted(checkpoints)[-1]
        best_path = os.path.join(checkpoint_dir, latest_checkpoint)

    print(f"Loading model from {best_path}")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)

    # Create model
    model = NL2CMTranslator(
        embedding_dim=1536,
        latent_dim=256,
        hidden_dim=512,
        adapter_depth=3,
        backbone_depth=4,
        dropout=0.1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def compute_baseline_metrics(nlt_emb: np.ndarray, cmt_emb: np.ndarray) -> dict:
    """Compute baseline metrics for comparison."""
    # Identity baseline (no translation)
    identity_cosine = np.mean([F.cosine_similarity(
        torch.FloatTensor(nlt_emb[i:i+1]),
        torch.FloatTensor(cmt_emb[i:i+1]),
        dim=1
    ).item() for i in range(len(nlt_emb))])

    # Random baseline
    n_samples = len(nlt_emb)
    random_mean_rank = n_samples / 2

    # Linear Procrustes baseline (orthogonal transformation)
    try:
        from scipy.linalg import orthogonal_procrustes
        R, _ = orthogonal_procrustes(cmt_emb, nlt_emb)
        nlt_procrustes = nlt_emb @ R.T
        procrustes_cosine = np.mean([F.cosine_similarity(
            torch.FloatTensor(nlt_procrustes[i:i+1]),
            torch.FloatTensor(cmt_emb[i:i+1]),
            dim=1
        ).item() for i in range(len(nlt_emb))])
    except ImportError:
        procrustes_cosine = 0.0

    return {
        'identity_cosine': identity_cosine,
        'random_mean_rank': random_mean_rank,
        'procrustes_cosine': procrustes_cosine
    }


def compute_retrieval_metrics(nlt_emb: np.ndarray, cmt_emb: np.ndarray,
                              translated_emb: np.ndarray) -> dict:
    """Compute comprehensive retrieval metrics."""
    # Compute similarities
    similarities = cosine_similarity(translated_emb, cmt_emb)

    # Get ranks for each query
    ranks = []
    for i in range(len(nlt_emb)):
        scores = similarities[i]
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)

    # Compute metrics
    metrics = {
        'mean_rank': np.mean(ranks),
        'median_rank': np.median(ranks),
        'top_1_accuracy': np.mean([1 if r == 1 else 0 for r in ranks]),
        'top_5_accuracy': np.mean([1 if r <= 5 else 0 for r in ranks]),
        'top_10_accuracy': np.mean([1 if r <= 10 else 0 for r in ranks]),
        'mrr': np.mean([1.0 / r for r in ranks])
    }

    return metrics


def create_evaluation_table(results: dict, baseline_results: dict, model_name: str = "NL2CM") -> str:
    """Create a comprehensive evaluation table."""
    table = "=" * 100 + "\n"
    table += f"Table: {model_name} Translation Evaluation Results\n"
    table += "=" * 100 + "\n\n"

    # Basic metrics
    table += f"{'Metric':<25} {'{model_name}':<15} {'Identity':<15} {'Procrustes':<15} {'Random':<15}\n"
    table += "-" * 85 + "\n"

    cosine_sim = results.get('cosine_similarity', 0.0)
    identity_cosine = baseline_results.get('identity_cosine', 0.0)
    procrustes_cosine = baseline_results.get('procrustes_cosine', 0.0)

    table += f"{'Cosine Similarity':<25} {cosine_sim:<15.4f} {identity_cosine:<15.4f} {procrustes_cosine:<15.4f} {'N/A':<15}\n"

    mean_rank = results.get('mean_rank', float('inf'))
    random_rank = baseline_results.get('random_mean_rank', float('inf'))

    table += f"{'Mean Rank':<25} {mean_rank:<15.2f} {'N/A':<15} {'N/A':<15} {random_rank:<15.2f}\n"

    top1 = results.get('top_1_accuracy', 0.0)
    top5 = results.get('top_5_accuracy', 0.0)
    top10 = results.get('top_10_accuracy', 0.0)
    mrr = results.get('mrr', 0.0)

    table += f"{'Top-1 Accuracy':<25} {top1:<15.4f} {'0.0000':<15} {'0.0000':<15} {'0.0000':<15}\n"
    table += f"{'Top-5 Accuracy':<25} {top5:<15.4f} {'0.0000':<15} {'0.0000':<15} {'0.0000':<15}\n"
    table += f"{'Top-10 Accuracy':<25} {top10:<15.4f} {'0.0000':<15} {'0.0000':<15} {'0.0000':<15}\n"
    table += f"{'MRR':<25} {mrr:<15.4f} {'0.0000':<15} {'0.0000':<15} {'0.0000':<15}\n\n"

    # Cycle consistency
    table += "Cycle Consistency:\n"
    table += "-" * 50 + "\n"
    table += f"{'NL Cycle Similarity':<25} {results.get('nlt_cycle_similarity', 0.0):<15.4f}\n"
    table += f"{'CM Cycle Similarity':<25} {results.get('cmt_cycle_similarity', 0.0):<15.4f}\n"
    table += f"{'Mean Cycle Similarity':<25} {results.get('mean_cycle_similarity', 0.0):<15.4f}\n\n"

    # Geometry preservation
    table += "Geometry Preservation:\n"
    table += "-" * 50 + "\n"
    table += f"{'NL Geometry Correlation':<25} {results.get('nlt_geometry_correlation', 0.0):<15.4f}\n"
    table += f"{'CM Geometry Correlation':<25} {results.get('cmt_geometry_correlation', 0.0):<15.4f}\n"
    table += f"{'Mean Geometry Correlation':<25} {results.get('mean_geometry_correlation', 0.0):<15.4f}\n\n"

    table += "=" * 100 + "\n"

    return table


def main():
    """Generate comprehensive evaluation tables."""
    print("Generating NL2CM Evaluation Tables")
    print("=" * 50)

    # Load data
    data_path = "datasets/eamodelset_nl2cm_embeddings_df.pkl"
    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    # Extract embeddings
    nlt_embeddings = np.stack(df['NL_Serialization_Emb'].values)
    cmt_embeddings = np.stack(df['CM_Serialization_Emb'].values)

    # Normalize embeddings
    nlt_embeddings = nlt_embeddings / \
        (np.linalg.norm(nlt_embeddings, axis=1, keepdims=True) + 1e-8)
    cmt_embeddings = cmt_embeddings / \
        (np.linalg.norm(cmt_embeddings, axis=1, keepdims=True) + 1e-8)

    print(f"Loaded {len(nlt_embeddings)} embeddings")

    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_best_model('checkpoints/nl2cm_full', device)

    # Create evaluator
    evaluator = NL2CMEvaluator(model, device)

    # Convert to tensors
    nlt_tensor = torch.FloatTensor(nlt_embeddings).to(device)
    cmt_tensor = torch.FloatTensor(cmt_embeddings).to(device)

    # Evaluate model
    print("Evaluating NL2CM model...")
    results = evaluator.evaluate(nlt_tensor, cmt_tensor)

    # Compute baseline metrics
    print("Computing baseline metrics...")
    baseline_results = compute_baseline_metrics(nlt_embeddings, cmt_embeddings)

    # Translate embeddings for additional metrics
    with torch.no_grad():
        translated_embeddings = model.translate_nlt_to_cmt(
            nlt_tensor).cpu().numpy()

    # Compute additional retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(
        nlt_embeddings, cmt_embeddings, translated_embeddings)
    results.update(retrieval_metrics)

    # Create evaluation table
    table = create_evaluation_table(results, baseline_results, "NL2CM")
    print("\n" + table)

    # Save results
    os.makedirs('evaluation_results', exist_ok=True)

    # Save table
    with open('evaluation_results/nl2cm_evaluation_table.txt', 'w') as f:
        f.write(table)

    # Save detailed results
    import json
    all_results = {
        'nl2cm_results': results,
        'baseline_results': baseline_results
    }

    with open('evaluation_results/nl2cm_detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Results saved to evaluation_results/")

    # Create a summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(
        f"Cosine Similarity: NL2CM={results['cosine_similarity']:.4f}, Identity={baseline_results['identity_cosine']:.4f}")
    print(
        f"Mean Rank: NL2CM={results['mean_rank']:.2f}, Random={baseline_results['random_mean_rank']:.2f}")
    print(f"Top-1 Accuracy: NL2CM={results['top_1_accuracy']:.4f}")
    print(f"MRR: NL2CM={results['mrr']:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
