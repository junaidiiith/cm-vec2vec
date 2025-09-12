"""
NL2CM Evaluation Module

This module implements evaluation metrics matching the vec2vec paper:
- Cosine similarity
- Top-1 accuracy  
- Mean rank
- Additional retrieval and classification metrics
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from nl2cm.utils import get_device
try:
    from .tensorboard_logger import NL2CMTensorBoardLogger
except ImportError:
    from tensorboard_logger import NL2CMTensorBoardLogger


class NL2CMEvaluator:
    """
    Evaluator for NL2CM translation model.

    This evaluator computes all metrics from the vec2vec paper and additional
    evaluation metrics for the NL2CM task.
    """

    def __init__(self, model: torch.nn.Module,
            use_tensorboard: bool = False, 
            tensorboard_logger: Optional[NL2CMTensorBoardLogger] = None
        ):
        """
        Initialize the evaluator.

        Args:
            model: The trained NL2CM model
            use_tensorboard: Whether to use TensorBoard logging
            tensorboard_logger: Optional TensorBoard logger instance
        """
        self.model = model.to(get_device())
        self.model.eval()
        self.use_tensorboard = use_tensorboard
        self.tensorboard_logger = tensorboard_logger

    def compute_cosine_similarity(self, nlt_emb: torch.Tensor,
                                  cmt_emb: torch.Tensor) -> float:
        """
        Compute mean cosine similarity between translated and target embeddings.

        Args:
            nlt_emb: NL embeddings (N, d)
            cmt_emb: CM embeddings (N, d)

        Returns:
            Mean cosine similarity
        """
        
        with torch.no_grad():
            # Translate NL to CM
            translated = self.model.translate_nlt_to_cmt(nlt_emb)

            # Compute cosine similarities
            similarities = F.cosine_similarity(translated, cmt_emb, dim=1)

            return similarities.mean().item()

    def compute_top_k_accuracy(self, nlt_emb: torch.Tensor, cmt_emb: torch.Tensor,
                               k: int = 1) -> float:
        """
        Compute Top-K accuracy for retrieval.

        Args:
            nlt_emb: NL embeddings (N, d)
            cmt_emb: CM embeddings (M, d) - can be different from N
            k: Number of top candidates to consider

        Returns:
            Top-K accuracy
        """
        with torch.no_grad():
            # Translate NL to CM
            translated = self.model.translate_nlt_to_cmt(nlt_emb)

            # Compute similarities with all CM embeddings
            similarities = torch.mm(translated, cmt_emb.t())

            # Get top-k indices for each query
            _, top_k_indices = torch.topk(similarities, k, dim=1)

            # Check if correct answer is in top-k
            correct = 0
            for i in range(len(nlt_emb)):
                if i < len(cmt_emb) and i in top_k_indices[i]:
                    correct += 1

            return correct / len(nlt_emb)

    def compute_mean_rank(self, nlt_emb: torch.Tensor, cmt_emb: torch.Tensor) -> float:
        """
        Compute mean rank of correct answers.

        Args:
            nlt_emb: NL embeddings (N, d)
            cmt_emb: CM embeddings (M, d) - can be different from N

        Returns:
            Mean rank
        """
        with torch.no_grad():
            # Translate NL to CM
            translated = self.model.translate_nlt_to_cmt(nlt_emb)

            # Compute similarities with all CM embeddings
            similarities = torch.mm(translated, cmt_emb.t())

            # Get ranks for each query
            ranks = []
            for i in range(len(nlt_emb)):
                if i < len(cmt_emb):
                    # Get similarity scores and sort in descending order
                    scores = similarities[i]
                    sorted_indices = torch.argsort(scores, descending=True)

                    # Find rank of correct answer
                    rank = (sorted_indices == i).nonzero(
                        as_tuple=True)[0].item() + 1
                    ranks.append(rank)

            return np.mean(ranks) if ranks else float('inf')

    def compute_retrieval_metrics(self, nlt_emb: torch.Tensor, cmt_emb: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive retrieval metrics.

        Args:
            nlt_emb: NL embeddings (N, d)
            cmt_emb: CM embeddings (M, d)

        Returns:
            Dictionary of retrieval metrics
        """
        with torch.no_grad():
            # Translate NL to CM
            translated = self.model.translate_nlt_to_cmt(nlt_emb)

            # Compute similarities
            similarities = torch.mm(translated, cmt_emb.t())

            # Get ranks for each query
            ranks = []
            for i in range(len(nlt_emb)):
                if i < len(cmt_emb):
                    scores = similarities[i]
                    sorted_indices = torch.argsort(scores, descending=True)
                    rank = (sorted_indices == i).nonzero(
                        as_tuple=True)[0].item() + 1
                    ranks.append(rank)

            # Compute metrics
            metrics = {}
            if ranks:
                metrics['mean_rank'] = np.mean(ranks)
                metrics['median_rank'] = np.median(ranks)
                metrics['top_1_accuracy'] = np.mean(
                    [1 if r == 1 else 0 for r in ranks])
                metrics['top_5_accuracy'] = np.mean(
                    [1 if r <= 5 else 0 for r in ranks])
                metrics['top_10_accuracy'] = np.mean(
                    [1 if r <= 10 else 0 for r in ranks])

                # MRR (Mean Reciprocal Rank)
                metrics['mrr'] = np.mean([1.0 / r for r in ranks])
            else:
                metrics = {
                    'mean_rank': float('inf'),
                    'median_rank': float('inf'),
                    'top_1_accuracy': 0.0,
                    'top_5_accuracy': 0.0,
                    'top_10_accuracy': 0.0,
                    'mrr': 0.0
                }

            return metrics

    def compute_cycle_consistency_metrics(self, nlt_emb: torch.Tensor,
                                          cmt_emb: torch.Tensor) -> Dict[str, float]:
        """
        Compute cycle consistency metrics.

        Args:
            nlt_emb: NL embeddings (N, d)
            cmt_emb: CM embeddings (N, d)

        Returns:
            Dictionary of cycle consistency metrics
        """
        with torch.no_grad():
            # Forward cycle: nlt -> cmt -> nlt
            nlt_to_cmt = self.model.translate_nlt_to_cmt(nlt_emb)
            nlt_cycle = self.model.translate_cmt_to_nlt(nlt_to_cmt)

            # Backward cycle: cmt -> nlt -> cmt
            cmt_to_nlt = self.model.translate_cmt_to_nlt(cmt_emb)
            cmt_cycle = self.model.translate_nlt_to_cmt(cmt_to_nlt)

            # Compute cycle consistency
            nlt_cycle_sim = F.cosine_similarity(
                nlt_emb, nlt_cycle, dim=1).mean().item()
            cmt_cycle_sim = F.cosine_similarity(
                cmt_emb, cmt_cycle, dim=1).mean().item()

            return {
                'nlt_cycle_similarity': nlt_cycle_sim,
                'cmt_cycle_similarity': cmt_cycle_sim,
                'mean_cycle_similarity': (nlt_cycle_sim + cmt_cycle_sim) / 2
            }

    def compute_geometry_preservation_metrics(self, nlt_emb: torch.Tensor,
                                              cmt_emb: torch.Tensor) -> Dict[str, float]:
        """
        Compute geometry preservation metrics.

        Args:
            nlt_emb: NL embeddings (N, d)
            cmt_emb: CM embeddings (N, d)

        Returns:
            Dictionary of geometry preservation metrics
        """
        with torch.no_grad():
            # Translate embeddings
            nlt_to_cmt = self.model.translate_nlt_to_cmt(nlt_emb)
            cmt_to_nlt = self.model.translate_cmt_to_nlt(cmt_emb)

            # Compute pairwise similarities
            nlt_orig_sim = torch.mm(nlt_emb, nlt_emb.t())
            nlt_trans_sim = torch.mm(nlt_to_cmt, nlt_to_cmt.t())

            cmt_orig_sim = torch.mm(cmt_emb, cmt_emb.t())
            cmt_trans_sim = torch.mm(cmt_to_nlt, cmt_to_nlt.t())

            # Compute correlation between original and translated similarities
            nlt_corr = torch.corrcoef(torch.stack([
                nlt_orig_sim.flatten(), nlt_trans_sim.flatten()
            ]))[0, 1].item()

            cmt_corr = torch.corrcoef(torch.stack([
                cmt_orig_sim.flatten(), cmt_trans_sim.flatten()
            ]))[0, 1].item()

            return {
                'nlt_geometry_correlation': nlt_corr,
                'cmt_geometry_correlation': cmt_corr,
                'mean_geometry_correlation': (nlt_corr + cmt_corr) / 2
            }

    def compute_classification_metrics(self, nlt_emb: torch.Tensor, cmt_emb: torch.Tensor,
                                       labels: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics using clustering.

        Args:
            nlt_emb: NL embeddings (N, d)
            cmt_emb: CM embeddings (N, d)
            labels: Ground truth labels (N,)

        Returns:
            Dictionary of classification metrics
        """
        with torch.no_grad():
            # Translate NL to CM
            translated = self.model.translate_nlt_to_cmt(nlt_emb)

            # Convert to numpy
            translated_np = translated.cpu().numpy()
            cmt_np = cmt_emb.cpu().numpy()

            # Perform K-means clustering
            n_clusters = len(np.unique(labels))

            # Cluster translated embeddings
            kmeans_trans = KMeans(n_clusters=n_clusters,
                                  random_state=42, n_init=10)
            trans_labels = kmeans_trans.fit_predict(translated_np)

            # Cluster original CM embeddings
            kmeans_orig = KMeans(n_clusters=n_clusters,
                                 random_state=42, n_init=10)
            orig_labels = kmeans_orig.fit_predict(cmt_np)

            # Compute metrics
            trans_ari = adjusted_rand_score(labels, trans_labels)
            trans_nmi = normalized_mutual_info_score(labels, trans_labels)

            orig_ari = adjusted_rand_score(labels, orig_labels)
            orig_nmi = normalized_mutual_info_score(labels, orig_labels)

            return {
                'translated_ari': trans_ari,
                'translated_nmi': trans_nmi,
                'original_ari': orig_ari,
                'original_nmi': orig_nmi,
                'ari_improvement': trans_ari - orig_ari,
                'nmi_improvement': trans_nmi - orig_nmi
            }

    def evaluate_data_loader(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a data loader.
        """
        batch_results = []
        for batch in tqdm(data_loader, desc="Evaluating Test Dataloader"):
            nlt_emb = batch['nlt']
            cmt_emb = batch['cmt']
            labels = batch['labels'] if 'labels' in batch else None
            if labels is not None:
                labels = labels.cpu().numpy()
            else:
                labels = None
            results = self.evaluate(nlt_emb, cmt_emb, labels)
            batch_results.append(results)
            
        results = {}
        for key in batch_results[0]:
            results[key] = np.mean([result[key] for result in batch_results])
        return results

    def evaluate(self, nlt_emb: torch.Tensor, cmt_emb: torch.Tensor,
                     labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            nlt_emb: NL embeddings (N, d)
            cmt_emb: CM embeddings (M, d)
            labels: Optional ground truth labels for classification metrics

        Returns:
            Dictionary of all metrics
        """
        results = {}

        # Basic translation metrics
        results['cosine_similarity'] = self.compute_cosine_similarity(nlt_emb.to(get_device()), cmt_emb.to(get_device()))

        # Retrieval metrics
        retrieval_metrics = self.compute_retrieval_metrics(nlt_emb.to(get_device()), cmt_emb.to(get_device()))
        results.update(retrieval_metrics)

        # Cycle consistency metrics
        cycle_metrics = self.compute_cycle_consistency_metrics(nlt_emb.to(get_device()), cmt_emb.to(get_device()))
        results.update(cycle_metrics)

        # Geometry preservation metrics
        geometry_metrics = self.compute_geometry_preservation_metrics(nlt_emb.to(get_device()), cmt_emb.to(get_device()))
        results.update(geometry_metrics)

        # Classification metrics (if labels provided)
        if labels is not None:
            classification_metrics = self.compute_classification_metrics(nlt_emb.to(get_device()), cmt_emb.to(get_device()), labels)
            results.update(classification_metrics)

        # Log to TensorBoard
        if self.tensorboard_logger:
            self.tensorboard_logger.log_evaluation_metrics(results, prefix="Final_Evaluation")

            # Log translation examples
            with torch.no_grad():
                translated_emb = self.model.translate_nlt_to_cmt(nlt_emb.to(get_device()))
                self.tensorboard_logger.log_translation_examples(nlt_emb.to(get_device()), cmt_emb.to(get_device()), translated_emb.to(get_device()))

        return results

    def create_evaluation_table(self, results: Dict[str, float]) -> str:
        """
        Create a formatted evaluation table similar to the vec2vec paper.

        Args:
            results: Dictionary of evaluation results

        Returns:
            Formatted table string
        """
        table = "=" * 80 + "\n"
        table += "NL2CM Translation Evaluation Results\n"
        table += "=" * 80 + "\n\n"

        # Basic metrics
        table += "Basic Translation Metrics:\n"
        table += "-" * 40 + "\n"
        table += f"Cosine Similarity: {results.get('cosine_similarity', 0.0):.4f}\n"
        table += f"Mean Rank: {results.get('mean_rank', float('inf')):.2f}\n"
        table += f"Top-1 Accuracy: {results.get('top_1_accuracy', 0.0):.4f}\n"
        table += f"Top-5 Accuracy: {results.get('top_5_accuracy', 0.0):.4f}\n"
        table += f"MRR: {results.get('mrr', 0.0):.4f}\n\n"

        # Cycle consistency
        table += "Cycle Consistency:\n"
        table += "-" * 40 + "\n"
        table += f"NL Cycle Similarity: {results.get('nlt_cycle_similarity', 0.0):.4f}\n"
        table += f"CM Cycle Similarity: {results.get('cmt_cycle_similarity', 0.0):.4f}\n"
        table += f"Mean Cycle Similarity: {results.get('mean_cycle_similarity', 0.0):.4f}\n\n"

        # Geometry preservation
        table += "Geometry Preservation:\n"
        table += "-" * 40 + "\n"
        table += f"NL Geometry Correlation: {results.get('nlt_geometry_correlation', 0.0):.4f}\n"
        table += f"CM Geometry Correlation: {results.get('cmt_geometry_correlation', 0.0):.4f}\n"
        table += f"Mean Geometry Correlation: {results.get('mean_geometry_correlation', 0.0):.4f}\n\n"

        # Classification (if available)
        if 'translated_ari' in results:
            table += "Classification Metrics:\n"
            table += "-" * 40 + "\n"
            table += f"Translated ARI: {results.get('translated_ari', 0.0):.4f}\n"
            table += f"Translated NMI: {results.get('translated_nmi', 0.0):.4f}\n"
            table += f"Original ARI: {results.get('original_ari', 0.0):.4f}\n"
            table += f"Original NMI: {results.get('original_nmi', 0.0):.4f}\n"
            table += f"ARI Improvement: {results.get('ari_improvement', 0.0):.4f}\n"
            table += f"NMI Improvement: {results.get('nmi_improvement', 0.0):.4f}\n\n"

        table += "=" * 80 + "\n"

        return table

    def plot_training_curves(self, train_losses: List[Dict], val_losses: List[Dict],
                             save_path: Optional[str] = None):
        """
        Plot training curves.

        Args:
            train_losses: List of training loss dictionaries
            val_losses: List of validation loss dictionaries
            save_path: Optional path to save the plot
        """
        epochs = range(1, len(train_losses) + 1)

        # Extract losses
        train_gen = [loss['total'] for loss in train_losses]
        train_disc = [loss['disc_total'] for loss in train_losses]
        val_gen = [loss['total'] for loss in val_losses]
        val_disc = [loss['disc_total'] for loss in val_losses]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Generator loss
        ax1.plot(epochs, train_gen, label='Train Generator', color='blue')
        ax1.plot(epochs, val_gen, label='Val Generator', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Generator Loss')
        ax1.set_title('Generator Loss')
        ax1.legend()
        ax1.grid(True)

        # Discriminator loss
        ax2.plot(epochs, train_disc, label='Train Discriminator', color='blue')
        ax2.plot(epochs, val_disc, label='Val Discriminator', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Discriminator Loss')
        ax2.set_title('Discriminator Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def save_results(self, results: Dict[str, float], save_path: str):
        """
        Save evaluation results to a file.

        Args:
            results: Dictionary of evaluation results
            save_path: Path to save the results
        """
        import json

        # Convert numpy types to Python types
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (np.integer, np.floating)):
                serializable_results[k] = v.item()
            else:
                serializable_results[k] = v

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {save_path}")
