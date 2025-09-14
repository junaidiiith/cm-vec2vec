import os
from typing import Optional, Dict
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)
from scipy.linalg import orthogonal_procrustes
from prettytable import PrettyTable


def get_device():
    """Get the device to use for training and evaluation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_baseline_metrics(source_emb: torch.Tensor, target_emb: torch.Tensor) -> dict:
    """Compute baseline metrics for comparison."""
    # Identity baseline (no translation)

    identity_cosine = np.mean([F.cosine_similarity(
        source_emb[i:i+1],
        target_emb[i:i+1],
        dim=1
    ).item() for i in range(len(source_emb))])

    # Random baseline
    n_samples = source_emb.shape[0]
    random_mean_rank = n_samples / 2

    # Linear Procrustes baseline (orthogonal transformation)
    R, _ = orthogonal_procrustes(
        target_emb.detach().cpu().numpy(), source_emb.detach().cpu().numpy())
    nlt_procrustes = source_emb.detach().cpu() @ R.T
    procrustes_cosine = np.mean([F.cosine_similarity(
        nlt_procrustes[i:i+1],
        target_emb[i:i+1].detach().cpu(),
        dim=1
    ).item() for i in range(source_emb.shape[0])])

    return {
        'identity_cosine': identity_cosine,
        'random_mean_rank': random_mean_rank,
        'procrustes_cosine': procrustes_cosine
    }


def compute_retrieval_metrics(source_emb: torch.Tensor, target_emb: torch.Tensor) -> dict:
    """Compute comprehensive retrieval metrics."""
    # Compute similarities
    cosine_similarity_results = compute_cosine_similarity(
        source_emb, target_emb)
    top_1_accuracy_results = compute_top_k_accuracy(
        source_emb, target_emb, k=1)
    top_5_accuracy_results = compute_top_k_accuracy(
        source_emb, target_emb, k=5)
    top_10_accuracy_results = compute_top_k_accuracy(
        source_emb, target_emb, k=10)
    mrr_results = compute_mrr(source_emb, target_emb)

    return {
        'cosine_similarity_results': cosine_similarity_results,
        'top_1_accuracy_results': top_1_accuracy_results,
        'top_5_accuracy_results': top_5_accuracy_results,
        'top_10_accuracy_results': top_10_accuracy_results,
        'mrr_results': mrr_results
    }


def compute_clustering_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute clustering-based evaluation metrics.

    Args:
        embeddings: Embeddings to cluster
        labels: Ground truth labels
        n_clusters: Number of clusters (if None, use unique labels)

    Returns:
        Dictionary of clustering metrics
    """
    with torch.no_grad():
        embeddings = embeddings.to(get_device())
        labels = labels.to(get_device())
        embeddings_np = embeddings.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()

        if n_clusters is None:
            n_clusters = len(np.unique(labels_np))

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)

        # Compute metrics
        ari = adjusted_rand_score(labels_np, cluster_labels)
        nmi = normalized_mutual_info_score(labels_np, cluster_labels)

        return {
            'ari': ari,
            'nmi': nmi
        }


def compute_cosine_similarity(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor
) -> float:
    """
    Compute mean cosine similarity between source and target embeddings.

    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings

    Returns:
        Mean cosine similarity
    """
    with torch.no_grad():
        source_embeddings = source_embeddings.to(get_device())
        target_embeddings = target_embeddings.to(get_device())

        # Normalize embeddings
        source_norm = F.normalize(source_embeddings, p=2, dim=1)
        target_norm = F.normalize(target_embeddings, p=2, dim=1)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(source_norm, target_norm, dim=1)

        return cosine_sim.mean().item()


def compute_top_k_accuracy(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    k: int = 1
) -> float:
    """
    Compute Top-K accuracy for retrieval.

    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings
        k: Number of top results to consider

    Returns:
        Top-K accuracy
    """
    with torch.no_grad():
        source_embeddings = source_embeddings.to(get_device())
        target_embeddings = target_embeddings.to(get_device())

        # Normalize embeddings
        source_norm = F.normalize(source_embeddings, p=2, dim=1)
        target_norm = F.normalize(target_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(source_norm, target_norm.t())

        # Get top-k indices
        _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)

        # Check if correct answer is in top-k
        correct = 0
        for i in range(len(source_embeddings)):
            if i in top_k_indices[i]:
                correct += 1

        return correct / len(source_embeddings)


def compute_mean_rank(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor
) -> float:
    """
    Compute mean rank of correct answers.

    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings

    Returns:
        Mean rank
    """
    with torch.no_grad():
        source_embeddings = source_embeddings.to(get_device())
        target_embeddings = target_embeddings.to(get_device())

        # Normalize embeddings
        source_norm = F.normalize(source_embeddings, p=2, dim=1)
        target_norm = F.normalize(target_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(source_norm, target_norm.t())

        # Get ranks
        ranks = []
        for i in range(len(source_embeddings)):
            # Get similarity scores for this source
            scores = similarity_matrix[i]
            # Sort in descending order and find rank of correct answer
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == i).nonzero(
                as_tuple=True)[0].item() + 1
            ranks.append(rank)

        return np.mean(ranks)


def compute_mrr(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings

    Returns:
        MRR score
    """
    with torch.no_grad():
        source_embeddings = source_embeddings.to(get_device())
        target_embeddings = target_embeddings.to(get_device())

        # Normalize embeddings
        source_norm = F.normalize(source_embeddings, p=2, dim=1)
        target_norm = F.normalize(target_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(source_norm, target_norm.t())

        # Get reciprocal ranks
        reciprocal_ranks = []
        for i in range(len(source_embeddings)):
            # Get similarity scores for this source
            scores = similarity_matrix[i]
            # Sort in descending order and find rank of correct answer
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == i).nonzero(
                as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)

        return np.mean(reciprocal_ranks)


def create_evaluation_table(results: dict, save_dir: str = "logs/cm_vec2vec") -> str:
    """
    Create a comprehensive evaluation table from flattened results dictionary.

    Args:
        results: Dictionary with flattened metric names as keys
        save_dir: Directory to save the table file

    Returns:
        Formatted table string
    """

    # Create the main table
    table = PrettyTable()
    table.field_names = ["Metric Category", "Metric", "Value"]
    table.align["Metric Category"] = "l"
    table.align["Metric"] = "l"
    table.align["Value"] = "r"

    # Group metrics by category based on key patterns
    for key, value in results.items():
        category = None
        metric_name = key

        # Cycle consistency metrics
        if any(x in key for x in ['cycle_similarity', 'mean_cycle']):
            category = "Cycle Consistency"
            if key == 'nlt_cycle_similarity':
                metric_name = "NLT Cycle Similarity"
            elif key == 'cmt_cycle_similarity':
                metric_name = "CMT Cycle Similarity"
            elif key == 'mean_cycle_similarity':
                metric_name = "Mean Cycle Similarity"

        # Geometry preservation metrics
        elif any(x in key for x in ['geometry_correlation', 'mean_geometry']):
            category = "Geometry Preservation"
            if key == 'nlt_geometry_correlation':
                metric_name = "NLT Geometry Correlation"
            elif key == 'cmt_geometry_correlation':
                metric_name = "CMT Geometry Correlation"
            elif key == 'mean_geometry_correlation':
                metric_name = "Mean Geometry Correlation"

        # Baseline metrics
        elif key.startswith('baseline_'):
            category = "Baseline"
            if 'nlt_to_cmt' in key:
                if 'identity_cosine' in key:
                    metric_name = "NLT→CMT Identity Cosine"
                elif 'random_mean_rank' in key:
                    metric_name = "NLT→CMT Random Mean Rank"
                elif 'procrustes_cosine' in key:
                    metric_name = "NLT→CMT Procrustes Cosine"
            elif 'cmt_to_nlt' in key:
                if 'identity_cosine' in key:
                    metric_name = "CMT→NLT Identity Cosine"
                elif 'random_mean_rank' in key:
                    metric_name = "CMT→NLT Random Mean Rank"
                elif 'procrustes_cosine' in key:
                    metric_name = "CMT→NLT Procrustes Cosine"

        # Retrieval metrics
        elif any(x in key for x in ['nlt2cmt', 'cmt2nlt', 'cosine_similarity', 'accuracy', 'mrr']):
            category = "Retrieval"
            if key.startswith('nlt2cmt_'):
                if 'cosine_similarity' in key:
                    metric_name = "NLT→CMT Cosine Similarity"
                elif 'top_1_accuracy' in key:
                    metric_name = "NLT→CMT Top-1 Accuracy"
                elif 'top_5_accuracy' in key:
                    metric_name = "NLT→CMT Top-5 Accuracy"
                elif 'top_10_accuracy' in key:
                    metric_name = "NLT→CMT Top-10 Accuracy"
                elif 'mrr' in key:
                    metric_name = "NLT→CMT MRR"
            elif key.startswith('cmt2nlt_'):
                if 'cosine_similarity' in key:
                    metric_name = "CMT→NLT Cosine Similarity"
                elif 'top_1_accuracy' in key:
                    metric_name = "CMT→NLT Top-1 Accuracy"
                elif 'top_5_accuracy' in key:
                    metric_name = "CMT→NLT Top-5 Accuracy"
                elif 'top_10_accuracy' in key:
                    metric_name = "CMT→NLT Top-10 Accuracy"
                elif 'mrr' in key:
                    metric_name = "CMT→NLT MRR"

        if category:
            # Handle numpy types
            if hasattr(value, 'item'):
                value = value.item()
            table.add_row([category, metric_name, f"{value:.4f}"])

    # Save table to file if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        table_file = os.path.join(save_dir, "evaluation_table.txt")
        with open(table_file, 'w') as f:
            f.write(str(table))
        print(f"Evaluation table saved to: {table_file}")

    return str(table)


def plot_embeddings(
    nlt_embeddings: torch.Tensor,
    cmt_embeddings: torch.Tensor,
    translated_nlt: Optional[torch.Tensor] = None,
    translated_cmt: Optional[torch.Tensor] = None,
    save_path: str = "logs/cm_vec2vec"
):
    """
    Plot embeddings using dimensionality reduction.

    Args:
        nlt_embeddings: Original NL embeddings
        cmt_embeddings: Original CM embeddings
        translated_nlt: Translated NL embeddings (optional)
        translated_cmt: Translated CM embeddings (optional)
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        save_path: Optional path to save the plot
    """

    with torch.no_grad():
        # Prepare data
        all_embeddings = []
        all_labels = []
        label_names = []

        # Add original embeddings
        nlt_np = nlt_embeddings.cpu().numpy()
        cmt_np = cmt_embeddings.cpu().numpy()

        all_embeddings.append(nlt_np)
        all_embeddings.append(cmt_np)
        all_labels.extend(['NLT'] * len(nlt_np))
        all_labels.extend(['CMT'] * len(cmt_np))
        label_names.extend(['Original NLT', 'Original CMT'])

        # Add translated embeddings if provided
        if translated_nlt is not None:
            trans_nlt_np = translated_nlt.cpu().numpy()
            all_embeddings.append(trans_nlt_np)
            all_labels.extend(['Translated NLT'] * len(trans_nlt_np))
            label_names.append('Translated NLT')

        if translated_cmt is not None:
            trans_cmt_np = translated_cmt.cpu().numpy()
            all_embeddings.append(trans_cmt_np)
            all_labels.extend(['Translated CMT'] * len(trans_cmt_np))
            label_names.append('Translated CMT')

        all_embeddings = np.vstack(all_embeddings)

        methods = {
            'tsne': TSNE(n_components=2, random_state=42),
            'pca': PCA(n_components=2, random_state=42),
            'umap': UMAP(n_components=2, random_state=42)
        }
        for method, reducer in methods.items():
            # Plot for all methods in subplots

            # Dimensionality reduction
            print(f"Dimensionality reduction with {method}")
            reduced_embeddings = reducer.fit_transform(all_embeddings)
            print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

            # Create plot
            plt.figure(figsize=(12, 8))
            unique_labels = list(set(all_labels))
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

            plt.title(f"Embeddings Visualization ({method.upper()})")
            plt.xlabel(f"{method.upper()} 1")
            plt.ylabel(f"{method.upper()} 2")

            plt.scatter(
                reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, label=label_names)
            # Plot by type
            plt.legend()
            plt.savefig(os.path.join(
                save_path, f"{method}.png"), dpi=300, bbox_inches='tight')
            plt.show()
