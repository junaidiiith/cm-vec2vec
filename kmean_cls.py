from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from vec2vec import run as train_vec2vec
from vec_transform import run as train_vecmap
import os
from utils import get_device
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


device = get_device()

def evaluate_with_kmeans(embeddings, true_labels, n_clusters):
    """
    Evaluate embeddings using K-means clustering and compare to true labels
    
    Args:
        embeddings: Normalized embedding vectors
        true_labels: Ground truth class labels (encoded)
        n_clusters: Number of clusters to create
        
    Returns:
        Dictionary of metrics
    """
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_clusters = kmeans.fit_predict(embeddings)
    
    # Calculate clustering metrics
    metrics = {
        "adjusted_rand_score": adjusted_rand_score(true_labels, predicted_clusters),
        "normalized_mutual_info": normalized_mutual_info_score(true_labels, predicted_clusters),
        "homogeneity_score": homogeneity_score(true_labels, predicted_clusters)
    }
    
    # Add intrinsic clustering metrics if the number of samples is sufficient
    if len(embeddings) > n_clusters:
        try:
            metrics["silhouette_score"] = silhouette_score(embeddings, predicted_clusters)
            metrics["davies_bouldin_score"] = davies_bouldin_score(embeddings, predicted_clusters)
        except Exception as e:
            print(f"Error calculating intrinsic metrics: {e}")
    
    return metrics, predicted_clusters




def train_model(X, Y, model_type='vec2vec'):
    """
    Train a model to map NL embeddings to CM embeddings
    
    Args:
        X: Normalized NL embeddings
        Y: Normalized CM embeddings
        model_type: 'vec2vec' or 'vecmap' to specify which model to train
        
    Returns:
        None (saves the trained model)
    """
    assert model_type in ['vec2vec', 'vecmap'], "model_type must be 'vec2vec' or 'vecmap'"
    
    if model_type == 'vec2vec':
        train_vec2vec(
            X=X, Y=Y,
            batch_size=64, epochs=100, lr=1e-4,
            d_lat=256, T=4, heads=4, layers=2, mem_tokens=4, dropout=0.1,
            save_path="runs/vec2vec_xattn.pt"
        )
    elif model_type == 'vecmap':
        train_vecmap(
            X=X, Y=Y,
            batch_size=64, epochs=100, lr=1e-4,
            d_lat=512, T_src=8, T_tgt=4,
            n_enc=3, n_dec=3, heads=8, mlp_ratio=4.0, dropout=0.1,
            save_path="runs/vecmap_transformer.pt"
        )


def get_predicted_embeddings(model_type, nl_embeddings):
    """
    Generate predicted CM embeddings from NL embeddings using trained models
    
    Args:
        model_type: 'vec2vec' or 'vecmap' to specify which model to use
        nl_embeddings: Normalized NL embeddings
        
    Returns:
        Predicted CM embeddings
    """
    assert model_type in ['vec2vec', 'vecmap'], "model_type must be 'vec2vec' or 'vecmap'"
    
    if model_type == 'vec2vec':
        # Load the vec2vec model
        assert os.path.exists("runs/vec2vec_xattn.pt"), "Trained vec2vec model not found at runs/vec2vec_xattn.pt"
        from vec2vec import Vec2VecXAttn
        
        d_text = nl_embeddings.shape[1]
        d_model = d_text  # Assuming same dimensions
        
        model = Vec2VecXAttn(
            d_text=d_text, 
            d_model=d_model,
            d_lat=256, T=4, heads=4, layers=2, mem_tokens=4, dropout=0.1
        ).to(device)
        
        model.load_state_dict(torch.load("runs/vec2vec_xattn.pt", map_location=device))
        model.eval()
        
        with torch.no_grad():
            x = torch.tensor(nl_embeddings, dtype=torch.float32, device=device)
            y_pred, _ = model.trans.text_to_model(x, y_hint=None)
            y_pred = F.normalize(y_pred, dim=-1).cpu().numpy()
        
    elif model_type == 'vecmap':
        # Load the vecmap model
        assert os.path.exists("runs/vecmap_transformer.pt"), "Trained vecmap model not found at runs/vecmap_transformer.pt"
        from vec_transform import VecMapTransformer
        
        d_src = nl_embeddings.shape[1]
        d_tgt = d_src  # Assuming same dimensions
        
        model = VecMapTransformer(
            d_src=d_src, d_tgt=d_tgt,
            d_lat=512, T_src=8, T_tgt=4,
            n_enc=3, n_dec=3, heads=8, mlp_ratio=4.0, dropout=0.1
        ).to(device)
        
        model.load_state_dict(torch.load("runs/vecmap_transformer.pt", map_location=device))
        model.eval()
        
        with torch.no_grad():
            x = torch.tensor(nl_embeddings, dtype=torch.float32, device=device)
            y_pred = model(x)
            y_pred = F.normalize(y_pred, dim=-1).cpu().numpy()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return y_pred


def evaluate_embeddings(X, Y_true, class_labels, n_clusters):
    # First, evaluate ground truth CM embeddings with K-means
    print("Evaluating ground truth CM embeddings...")
    cm_metrics, cm_clusters = evaluate_with_kmeans(Y_true, class_labels, n_clusters)
    print("Ground truth CM embeddings results:")
    for metric, value in cm_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Generate predicted embeddings using vec2vec model
    try:
        print("\nGenerating vec2vec predicted embeddings...")
        vec2vec_pred = get_predicted_embeddings('vec2vec', X)
        
        # Evaluate vec2vec predicted embeddings
        print("Evaluating vec2vec predicted embeddings...")
        vec2vec_metrics, vec2vec_clusters = evaluate_with_kmeans(vec2vec_pred, class_labels, n_clusters)
        print("vec2vec predicted embeddings results:")
        for metric, value in vec2vec_metrics.items():
            print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error evaluating vec2vec model: {e}")

    # Generate predicted embeddings using vecmap model
    try:
        print("\nGenerating vecmap predicted embeddings...")
        vecmap_pred = get_predicted_embeddings('vecmap', X)
        
        # Evaluate vecmap predicted embeddings
        print("Evaluating vecmap predicted embeddings...")
        vecmap_metrics, vecmap_clusters = evaluate_with_kmeans(vecmap_pred, class_labels, n_clusters)
        print("vecmap predicted embeddings results:")
        for metric, value in vecmap_metrics.items():
            print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error evaluating vecmap model: {e}")
        

# Visualize the comparison of metrics
def plot_metrics_comparison(cm_metrics, vec2vec_metrics, vecmap_metrics):
    # Collect metrics for comparison
    metrics = list(cm_metrics.keys())
    values = {
        'Ground Truth CM': [cm_metrics[m] for m in metrics]
    }
    
    if vec2vec_metrics is not None:
        values['vec2vec Predicted'] = [vec2vec_metrics.get(m, 0) for m in metrics]
    
    if vecmap_metrics is not None:
        values['vecmap Predicted'] = [vecmap_metrics.get(m, 0) for m in metrics]
    
    # Plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.2
    multiplier = 0
    
    for model, scores in values.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, scores, width, label=model)
        multiplier += 1
    
    plt.ylabel('Score')
    plt.title('Comparison of Clustering Metrics')
    plt.xticks(x + width, metrics, rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    
    # Add value annotations on top of bars
    for i, model in enumerate(values.keys()):
        for j, v in enumerate(values[model]):
            plt.text(j + width * (i - 0.5), v + 0.01, f'{v:.2f}', 
                     ha='center', va='bottom', fontsize=8)
    
    plt.ylim(0, 1.0)  # All metrics are between 0 and 1
    plt.show()

    try:
        # Create the comparison visualization
        plot_metrics_comparison(
            cm_metrics, 
            vec2vec_metrics,
            vecmap_metrics
        )
    except Exception as e:
        print(f"Error creating visualization: {e}")
        


def plot_tsne_clusters(X, Y_true, Y_Pred, labels, title, n_clusters):
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    reduced_data = tsne.fit_transform(X)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, 
                          cmap='tab20', alpha=0.8, s=50)
    
    # Add a legend with class names
    class_names = [f"Class {i}" for i in range(n_clusters)]
    legend1 = plt.legend(handles=scatter.legend_elements()[0], 
                        labels=class_names,
                        title="Classes", loc="upper right")
    plt.gca().add_artist(legend1)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # Visualize clusters for all embeddings
    try:
        print("\nVisualizing clusters using t-SNE...")
        plot_tsne_clusters(Y_true, labels, "Ground Truth CM Embeddings")
        plot_tsne_clusters(Y_Pred, labels, "Predicted CM Embeddings")
    except Exception as e:
        print(f"Error creating t-SNE visualizations: {e}")