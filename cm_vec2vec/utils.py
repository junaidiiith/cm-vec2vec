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
from scipy.optimize import linear_sum_assignment
from prettytable import PrettyTable
from tqdm.auto import tqdm


def get_device():
    """Get the device to use for training and evaluation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_baseline_metrics(original_source_emb: np.ndarray, original_target_emb: np.ndarray) -> dict:
    """
    Compute baseline metrics for comparison using original embeddings.

    Args:
        original_source_emb: Original source embeddings (e.g., NLT embeddings)
        original_target_emb: Original target embeddings (e.g., CMT embeddings)

    Returns:
        Dictionary of baseline metrics
    """
    # Ensure inputs are numpy arrays
    original_source_emb = np.asarray(original_source_emb)
    original_target_emb = np.asarray(original_target_emb)

    # 1. Identity baseline (F(x) = x) - no translation
    # This tests how similar the original embeddings are to each other
    identity_cosine = np.mean([
        compute_cosine_similarity_torch(
            original_source_emb[i:i+1], original_target_emb[i:i+1])
        for i in range(len(original_source_emb))
    ])

    # 2. Random baseline - expected rank for random assignment
    n_samples = original_source_emb.shape[0]
    random_mean_rank = n_samples / 2

    # 3. Procrustes baseline (orthogonal transformation on original embeddings)
    # This finds the best linear transformation between original embedding spaces
    R, _ = orthogonal_procrustes(original_target_emb, original_source_emb)
    procrustes_transformed = original_source_emb @ R.T
    procrustes_cosine = np.mean([
        compute_cosine_similarity_torch(
            procrustes_transformed[i:i+1], original_target_emb[i:i+1])
        for i in range(original_source_emb.shape[0])
    ])

    # 4. Identity baseline retrieval metrics (F(x) = x)
    print("Computing identity baseline retrieval metrics...")
    identity_mrr = compute_mrr_torch(original_source_emb, original_target_emb)
    identity_top_1 = compute_top_k_accuracy_torch(
        original_source_emb, original_target_emb, k=1)
    identity_top_5 = compute_top_k_accuracy_torch(
        original_source_emb, original_target_emb, k=5)
    identity_top_10 = compute_top_k_accuracy_torch(
        original_source_emb, original_target_emb, k=10)

    # 5. Procrustes baseline retrieval metrics
    print("Computing procrustes baseline retrieval metrics...")
    procrustes_mrr = compute_mrr_torch(
        procrustes_transformed, original_target_emb)
    procrustes_top_1 = compute_top_k_accuracy_torch(
        procrustes_transformed, original_target_emb, k=1)
    procrustes_top_5 = compute_top_k_accuracy_torch(
        procrustes_transformed, original_target_emb, k=5)
    procrustes_top_10 = compute_top_k_accuracy_torch(
        procrustes_transformed, original_target_emb, k=10)

    # 6. Oracle-aided optimal transport baseline
    # print("Computing oracle-aided optimal transport baseline...")
    # oracle_mrr, oracle_top_1, oracle_top_5, oracle_top_10 = compute_oracle_baseline(
    #     original_source_emb, original_target_emb)

    return {
        'identity_cosine': identity_cosine,
        'random_mean_rank': random_mean_rank,
        'procrustes_cosine': procrustes_cosine,
        'identity_mrr': identity_mrr,
        'identity_top_1_accuracy': identity_top_1,
        'identity_top_5_accuracy': identity_top_5,
        'identity_top_10_accuracy': identity_top_10,
        'procrustes_mrr': procrustes_mrr,
        'procrustes_top_1_accuracy': procrustes_top_1,
        'procrustes_top_5_accuracy': procrustes_top_5,
        'procrustes_top_10_accuracy': procrustes_top_10,
        # 'oracle_mrr': oracle_mrr,
        # 'oracle_top_1_accuracy': oracle_top_1,
        # 'oracle_top_5_accuracy': oracle_top_5,
        # 'oracle_top_10_accuracy': oracle_top_10
    }


def compute_oracle_baseline(source_emb: np.ndarray, target_emb: np.ndarray) -> tuple:
    """
    Compute oracle-aided optimal transport baseline.

    This solves the optimal assignment problem: π* = arg min_π Σ cos(u_i, v_π(i))
    This is the best possible performance when we know all candidate targets.

    Args:
        source_emb: Source embeddings
        target_emb: Target embeddings

    Returns:
        Tuple of (mrr, top_1, top_5, top_10) for oracle baseline
    """
    # Normalize embeddings
    print("Computing cosine similarity matrix...")
    similarity_matrix = fast_pairwise_cosine_similarity(source_emb, target_emb)
    print("Cosine similarity matrix computed successfully of shape", similarity_matrix.shape)
    # Solve optimal assignment problem using Hungarian algorithm
    # We want to maximize cosine similarity, so negate the matrix
    print("Solving optimal assignment problem...")
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    print("Optimal assignment problem solved successfully!")

    # Create optimal assignment mapping
    optimal_assignment = np.zeros(len(source_emb), dtype=int)
    optimal_assignment[row_indices] = col_indices

    # Compute metrics for optimal assignment
    # For each source, find its rank in the similarity to its assigned target
    ranks = []
    for i in tqdm(range(len(source_emb)), desc="Computing ranks"):
        # Get similarity scores for source i to all targets
        similarities = similarity_matrix[i]
        # Find rank of the optimally assigned target
        assigned_target = optimal_assignment[i]
        assigned_similarity = similarities[assigned_target]
        # Count how many targets have higher similarity
        rank = np.sum(similarities > assigned_similarity) + 1
        ranks.append(rank)
    print("Ranks computed successfully!")
    ranks = np.array(ranks)

    # Compute metrics
    mrr = np.mean(1.0 / ranks)
    top_1 = np.mean(ranks == 1)
    top_5 = np.mean(ranks <= 5)
    top_10 = np.mean(ranks <= 10)
    print("Metrics computed successfully!")
    return mrr, top_1, top_5, top_10


def compute_retrieval_metrics(source_emb: np.ndarray, target_emb: np.ndarray) -> dict:
    """Compute comprehensive retrieval metrics."""
    print("Computing comprehensive retrieval metrics...")
    # Compute similarities
    cosine_similarity_results = compute_cosine_similarity_torch(
        source_emb, target_emb)
    top_1_accuracy_results = compute_top_k_accuracy_torch(
        source_emb, target_emb, k=1)
    top_5_accuracy_results = compute_top_k_accuracy_torch(
        source_emb, target_emb, k=5)
    top_10_accuracy_results = compute_top_k_accuracy_torch(
        source_emb, target_emb, k=10)
    mrr_results = compute_mrr_torch(source_emb, target_emb)

    return {
        'cosine_similarity_results': cosine_similarity_results,
        'top_1_accuracy_results': top_1_accuracy_results,
        'top_5_accuracy_results': top_5_accuracy_results,
        'top_10_accuracy_results': top_10_accuracy_results,
        'mrr_results': mrr_results
    }


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
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

    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute metrics
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)

    return {
        'ari': ari,
        'nmi': nmi
    }


def compute_cosine_similarity(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray
) -> float:
    """
    Compute mean cosine similarity between source and target embeddings.

    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings

    Returns:
        Mean cosine similarity
    """

    # Normalize vectors to unit length
    source_norm = source_embeddings / \
        np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    target_norm = target_embeddings / \
        np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    # Compute cosine similarity as dot product of normalized vectors
    cosine_sim = np.sum(source_norm * target_norm, axis=1)

    return np.mean(cosine_sim)


def compute_cosine_loss(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray
) -> float:
    """
    Compute cosine loss as defined in the paper: (1/n) * Σ [1 - cos(F(u_i), v_i)]

    This is the loss function used for training the translator.

    Args:
        source_embeddings: Translated source embeddings F(u_i)
        target_embeddings: Target embeddings v_i

    Returns:
        Mean cosine loss
    """
    # Normalize vectors to unit length
    source_norm = source_embeddings / \
        np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    target_norm = target_embeddings / \
        np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    # Compute cosine similarity as dot product of normalized vectors
    cosine_sim = np.sum(source_norm * target_norm, axis=1)

    # Compute cosine loss: 1 - cosine_similarity
    cosine_loss = 1.0 - cosine_sim

    return np.mean(cosine_loss)


def compute_cosine_similarity_torch(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    batch_size: int = 2000,
    device: str = 'cuda'
) -> float:
    """
    GPU-accelerated cosine similarity computation.
    
    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Mean cosine similarity
    """
    # Convert to PyTorch tensors
    source_tensor = torch.tensor(source_embeddings, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_embeddings, dtype=torch.float32, device=device)
    
    # Normalize vectors to unit length
    source_norm = F.normalize(source_tensor, p=2, dim=1)
    target_norm = F.normalize(target_tensor, p=2, dim=1)
    
    n_samples = len(source_embeddings)
    cosine_similarities = []
    
    with torch.no_grad():
        # print("Computing cosine similarities...")
        for i in range(0, n_samples, batch_size):
        # for i in tqdm(range(0, n_samples, batch_size), desc="Computing cosine similarities"):
            end_i = min(i + batch_size, n_samples)
            
            # Get batch
            batch_source = source_norm[i:end_i]
            batch_target = target_norm[i:end_i]
            
            # Compute cosine similarity as dot product of normalized vectors
            batch_cosine_sim = torch.sum(batch_source * batch_target, dim=1)
            cosine_similarities.append(batch_cosine_sim.cpu().numpy())

    # Concatenate and compute mean
    all_cosine_sim = np.concatenate(cosine_similarities)
    return np.mean(all_cosine_sim)


def compute_top_k_accuracy(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    k: int = 1
) -> float:
    """
    Vectorized Top-K accuracy computation.
    """
    source_embeddings = np.asarray(source_embeddings)
    target_embeddings = np.asarray(target_embeddings)

    # Normalize embeddings
    source_norm = source_embeddings / \
        np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    target_norm = target_embeddings / \
        np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = np.dot(source_norm, target_norm.T)

    # Get top-k indices
    top_k_indices = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]

    # Check if diagonal elements (correct answers) are in top-k
    diagonal_indices = np.arange(len(source_embeddings))[:, np.newaxis]
    correct = np.any(top_k_indices == diagonal_indices, axis=1)

    return np.mean(correct)


def compute_top_k_accuracy_torch(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    k: int = 1,
    batch_size: int = 2000,
    device: str = 'cuda'
) -> float:
    """
    Ultra-fast Top-K accuracy using PyTorch GPU acceleration.
    """
    # Convert to PyTorch tensors
    source_tensor = torch.tensor(
        source_embeddings, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(
        target_embeddings, dtype=torch.float32, device=device)

    # Normalize embeddings
    source_norm = F.normalize(source_tensor, p=2, dim=1)
    target_norm = F.normalize(target_tensor, p=2, dim=1)

    n_samples = len(source_embeddings)
    correct_count = 0

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Computing top-k accuracy for k={k}"):
            end_i = min(i + batch_size, n_samples)

            # Get batch
            batch_source = source_norm[i:end_i]

            # Compute similarities
            similarities = torch.mm(batch_source, target_norm.t())

            # Get top-k indices
            _, top_k_indices = torch.topk(similarities, k, dim=1)

            # Check correctness
            correct_indices = torch.arange(
                i, end_i, device=device).unsqueeze(1)
            batch_correct = torch.any(top_k_indices == correct_indices, dim=1)

            correct_count += batch_correct.sum().item()

    return correct_count / n_samples


def compute_mean_rank(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray
) -> float:
    """
    Vectorized mean rank computation.
    """
    source_embeddings = np.asarray(source_embeddings)
    target_embeddings = np.asarray(target_embeddings)

    # Normalize embeddings
    source_norm = source_embeddings / \
        np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    target_norm = target_embeddings / \
        np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = np.dot(source_norm, target_norm.T)

    # Get ranks for all queries at once
    sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    ranks = np.argmax(sorted_indices == np.arange(
        len(source_embeddings))[:, np.newaxis], axis=1) + 1

    return np.mean(ranks)


def compute_mrr(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray
) -> float:
    """
    Vectorized MRR computation.
    """
    source_embeddings = np.asarray(source_embeddings)
    target_embeddings = np.asarray(target_embeddings)

    # Normalize embeddings
    source_norm = source_embeddings / \
        np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    target_norm = target_embeddings / \
        np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = np.dot(source_norm, target_norm.T)

    # Get ranks for all queries at once
    sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    ranks = np.argmax(sorted_indices == np.arange(
        len(source_embeddings))[:, np.newaxis], axis=1) + 1

    # Compute reciprocal ranks
    reciprocal_ranks = 1.0 / ranks

    return np.mean(reciprocal_ranks)


def compute_mrr_torch(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    batch_size: int = 2000,
    device: str = 'cuda'
) -> float:
    """
    Ultra-fast MRR computation using PyTorch GPU acceleration.
    Memory-efficient version that avoids sorting large similarity matrices.
    """
    # Convert to PyTorch tensors
    source_tensor = torch.tensor(
        source_embeddings, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(
        target_embeddings, dtype=torch.float32, device=device)

    # Normalize embeddings
    source_norm = F.normalize(source_tensor, p=2, dim=1)
    target_norm = F.normalize(target_tensor, p=2, dim=1)

    n_samples = len(source_embeddings)
    reciprocal_ranks = []

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Computing MRR"):
            end_i = min(i + batch_size, n_samples)

            # Get batch
            batch_source = source_norm[i:end_i]

            # Compute similarities
            similarities = torch.mm(batch_source, target_norm.t())

            # Memory-efficient rank computation
            # For each query, find how many targets have higher similarity than the correct answer
            correct_indices = torch.arange(i, end_i, device=device)
            correct_similarities = similarities[torch.arange(
                end_i - i), correct_indices]

            # Count how many targets have higher similarity than correct answer
            # This gives us the rank directly without sorting
            ranks = (similarities > correct_similarities.unsqueeze(1)).sum(
                dim=1) + 1

            # Compute reciprocal ranks
            batch_reciprocal_ranks = 1.0 / ranks.float()
            reciprocal_ranks.extend(batch_reciprocal_ranks.cpu().numpy())

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
    nlt_embeddings: np.ndarray,
    cmt_embeddings: np.ndarray,
    translated_nlt: Optional[np.ndarray] = None,
    translated_cmt: Optional[np.ndarray] = None,
    save_path: str = "logs/cm_vec2vec",
    methods: Optional[list] = None,
    figsize: tuple = (15, 10)
):
    """
    Enhanced plot embeddings with better visualization options.

    Args:
        nlt_embeddings: Original NL embeddings (numpy array)
        cmt_embeddings: Original CM embeddings (numpy array)
        translated_nlt: Translated NL embeddings (optional, numpy array)
        translated_cmt: Translated CM embeddings (optional, numpy array)
        save_path: Optional path to save the plot
        methods: List of methods to use ['tsne', 'pca', 'umap'] (default: all)
        figsize: Figure size tuple
    """
    # Ensure inputs are numpy arrays
    nlt_embeddings = np.asarray(nlt_embeddings)
    cmt_embeddings = np.asarray(cmt_embeddings)

    if methods is None:
        methods = ['tsne', 'pca', 'umap']

    # Prepare data
    all_embeddings = []
    all_labels = []
    label_names = []

    # Add original embeddings
    all_embeddings.append(nlt_embeddings)
    all_embeddings.append(cmt_embeddings)
    all_labels.extend(['NLT'] * len(nlt_embeddings))
    all_labels.extend(['CMT'] * len(cmt_embeddings))
    label_names.extend(['Original NLT', 'Original CMT'])

    # Add translated embeddings if provided
    if translated_nlt is not None:
        translated_nlt = np.asarray(translated_nlt)
        all_embeddings.append(translated_nlt)
        all_labels.extend(['Translated NLT'] * len(translated_nlt))
        label_names.append('Translated NLT')

    if translated_cmt is not None:
        translated_cmt = np.asarray(translated_cmt)
        all_embeddings.append(translated_cmt)
        all_labels.extend(['Translated CMT'] * len(translated_cmt))
        label_names.append('Translated CMT')

    # Stack all embeddings
    all_embeddings = np.vstack(all_embeddings)

    # Define dimensionality reduction methods
    reducers = {
        'tsne': TSNE(n_components=2, random_state=42, perplexity=30),
        'pca': PCA(n_components=2, random_state=42),
        'umap': UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    }

    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Create subplots if multiple methods
    if len(methods) > 1:
        fig, axes = plt.subplots(1, len(methods), figsize=figsize)
        if len(methods) == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]

    for idx, method in enumerate(methods):
        if method not in reducers:
            print(f"Warning: Method {method} not supported. Skipping.")
            continue

        reducer = reducers[method]
        ax = axes[idx] if len(methods) > 1 else axes[0]

        # Dimensionality reduction
        print(f"Dimensionality reduction with {method}")
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

        # Define colors and markers for different types
        unique_labels = list(set(all_labels))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

        # Plot by type with proper color and marker mapping
        for i, label in enumerate(unique_labels):
            mask = np.array(all_labels) == label
            ax.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7,
                s=50,
                marker=markers[i % len(markers)],
                edgecolors='black',
                linewidth=0.5
            )

        ax.set_title(
            f"Embeddings Visualization ({method.upper()})", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{method.upper()} 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if len(methods) > 1:
        plt.savefig(os.path.join(save_path, "embeddings_comparison.png"),
                    dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(
            save_path, f"{methods[0]}.png"), dpi=300, bbox_inches='tight')

    plt.show()


def fast_pairwise_cosine_similarity(embeddings1, embeddings2, chunk_size=1000):
    """
    Compute pairwise cosine similarity matrix efficiently.

    Args:
        embeddings1: (N, D) array
        embeddings2: (M, D) array
        chunk_size: Size of chunks for memory efficiency

    Returns:
        (N, M) similarity matrix
    """
    # Normalize embeddings
    emb1_norm = embeddings1 / \
        np.linalg.norm(embeddings1, axis=1, keepdims=True)
    emb2_norm = embeddings2 / \
        np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Compute similarity matrix in chunks
    similarities = []
    # for i in tqdm(range(0, len(emb1_norm), chunk_size), desc="Computing pairwise cosine similarity"):
    for i in range(0, len(emb1_norm), chunk_size):
        end_i = min(i + chunk_size, len(emb1_norm))
        chunk_sim = np.dot(emb1_norm[i:end_i], emb2_norm.T)
        similarities.append(chunk_sim)

    return np.vstack(similarities)


def compute_geometric_preservation_score(source_embeddings, target_embeddings, batch_size=1000):
    """
    Compute a normalized geometric preservation score (0-1, higher is better).

    This measures the correlation between original and translated pairwise similarities.
    """
    n_samples = len(source_embeddings)
    all_orig_sims = []
    all_trans_sims = []

    # Process in batches
    # for i in tqdm(range(0, n_samples, batch_size), desc="Computing geometric preservation score"):
    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        # Normalize embeddings
        batch_orig_norm = source_embeddings[i:end_i] / np.linalg.norm(
            source_embeddings[i:end_i], axis=1, keepdims=True)
        batch_trans_norm = target_embeddings[i:end_i] / np.linalg.norm(
            target_embeddings[i:end_i], axis=1, keepdims=True)

        # Compute cosine similarities
        orig_sims = np.dot(batch_orig_norm, batch_orig_norm.T)
        trans_sims = np.dot(batch_trans_norm, target_embeddings[i:end_i].T)

        # Extract upper triangular part (excluding diagonal) to avoid duplicates
        mask = np.triu(np.ones_like(orig_sims, dtype=bool), k=1)
        all_orig_sims.extend(orig_sims[mask])
        all_trans_sims.extend(trans_sims[mask])

    # Compute correlation between original and translated similarities
    if len(all_orig_sims) > 1:
        correlation = np.corrcoef(all_orig_sims, all_trans_sims)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    else:
        return 0.0


def compute_cycle_consistency_score(original_embeddings, cycled_embeddings):
    """
    Compute normalized cycle consistency score (0-1, higher is better).

    This is a normalized version where 1.0 means perfect reconstruction.
    """
    # Compute L2 distances
    diff = original_embeddings - cycled_embeddings
    l2_distances = np.linalg.norm(diff, axis=1)

    # Normalize by the magnitude of original embeddings
    original_magnitudes = np.linalg.norm(original_embeddings, axis=1)

    # Avoid division by zero
    normalized_distances = np.where(
        original_magnitudes > 1e-8,
        l2_distances / original_magnitudes,
        l2_distances
    )

    # Convert to similarity score (1.0 = perfect, 0.0 = worst)
    # Using exponential decay: score = exp(-normalized_distance)
    scores = np.exp(-normalized_distances)

    return np.mean(scores)
