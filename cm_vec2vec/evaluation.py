"""
Evaluation utilities for CMVec2Vec
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, Optional, Any, Union
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from cm_vec2vec.translators.cm_vec2vec_translator import CMVec2VecTranslator
from cm_vec2vec.utils import get_device


class CMVec2VecEvaluator:
    """
    Evaluator class for CMVec2Vec model.

    Provides comprehensive evaluation metrics including translation quality,
    cycle consistency, geometry preservation, and clustering metrics.
    """

    def __init__(self, model: CMVec2VecTranslator, save_dir: str = 'logs/cm_vec2vec'):
        """
        Initialize the evaluator.

        Args:
            model: Trained CMVec2Vec model
        """
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()
        self.save_dir = save_dir
        self.writer = SummaryWriter(save_dir)
    
    def _get_batch_embeddings(self, batch: Dict[str, torch.Tensor], key: str, translate_fn: Callable) -> torch.Tensor:
            condition = batch.get('condition', None)
            if isinstance(batch, dict):
                condition = batch.get('condition', None)
            if condition is not None:
                condition = condition.to(self.device)
            return translate_fn(batch[key].to(self.device), condition)
    
    
    def get_nlt_to_cmt_embeddings(self, batch: Union[Dict[str, torch.Tensor], DataLoader]) -> torch.Tensor:
        """
        Get embeddings from a batch.
        """
        if isinstance(batch, dict):
            return self._get_batch_embeddings(batch, 'nlt', self.model.translate_nlt_to_cmt)
        elif isinstance(batch, DataLoader):
            return torch.cat([self._get_batch_embeddings(batch, 'nlt', self.model.translate_nlt_to_cmt) for batch in batch], dim=0)
        else:
            raise ValueError(f"Invalid batch type: {type(batch)}")
    

    def get_cmt_to_nlt_embeddings(self, batch: Union[Dict[str, torch.Tensor], DataLoader]) -> torch.Tensor:
        """
        Get embeddings from a batch.    
        """
        if isinstance(batch, dict):
            return self._get_batch_embeddings(batch, 'cmt', self.model.translate_cmt_to_nlt)
        elif isinstance(batch, DataLoader):
            return torch.cat([self._get_batch_embeddings(batch, 'cmt', self.model.translate_cmt_to_nlt) for batch in batch], dim=0)
        else:
            raise ValueError(f"Invalid batch type: {type(batch)}")


    def compute_cosine_similarity(
        self,
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
            source_embeddings = source_embeddings.to(self.device)
            target_embeddings = target_embeddings.to(self.device)

            # Normalize embeddings
            source_norm = F.normalize(source_embeddings, p=2, dim=1)
            target_norm = F.normalize(target_embeddings, p=2, dim=1)

            # Compute cosine similarity
            cosine_sim = F.cosine_similarity(source_norm, target_norm, dim=1)

            return cosine_sim.mean().item()

    def compute_top_k_accuracy(
        self,
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
            source_embeddings = source_embeddings.to(self.device)
            target_embeddings = target_embeddings.to(self.device)

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
        self,
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
            source_embeddings = source_embeddings.to(self.device)
            target_embeddings = target_embeddings.to(self.device)

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
        self,
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
            source_embeddings = source_embeddings.to(self.device)
            target_embeddings = target_embeddings.to(self.device)

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

    def compute_cycle_consistency_metrics(
        self,
        nlt_embeddings: torch.Tensor,
        cmt_embeddings: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute cycle consistency metrics.

        Args:
            nlt_embeddings: NL embeddings
            cmt_embeddings: CM embeddings
            condition: Optional conditioning vector

        Returns:
            Dictionary of cycle consistency metrics
        """
        with torch.no_grad():
            nlt_embeddings = nlt_embeddings.to(self.device)
            cmt_embeddings = cmt_embeddings.to(self.device)
            if condition is not None:
                condition = condition.to(self.device)

            # Forward translation: NL -> CM
            forward_translated = self.model.translate_nlt_to_cmt(
                nlt_embeddings, condition
            )

            # Backward translation: CM -> NL
            backward_translated = self.model.translate_cmt_to_nlt(
                cmt_embeddings, condition
            )

            # Cycle consistency: NL -> CM -> NL
            cycle_nlt = self.model.translate_cmt_to_nlt(
                forward_translated, condition
            )

            # Cycle consistency: CM -> NL -> CM
            cycle_cmt = self.model.translate_nlt_to_cmt(
                backward_translated, condition
            )

            # Compute cycle similarities
            nlt_cycle_sim = self.compute_cosine_similarity(
                nlt_embeddings, cycle_nlt)
            cmt_cycle_sim = self.compute_cosine_similarity(
                cmt_embeddings, cycle_cmt)

            return {
                'nlt_cycle_similarity': nlt_cycle_sim,
                'cmt_cycle_similarity': cmt_cycle_sim,
                'mean_cycle_similarity': (nlt_cycle_sim + cmt_cycle_sim) / 2
            }

    def compute_geometry_preservation_metrics(
        self,
        nlt_embeddings: torch.Tensor,
        cmt_embeddings: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute geometry preservation metrics.

        Args:
            nlt_embeddings: NL embeddings
            cmt_embeddings: CM embeddings
            condition: Optional conditioning vector

        Returns:
            Dictionary of geometry preservation metrics
        """
        with torch.no_grad():
            nlt_embeddings = nlt_embeddings.to(self.device)
            cmt_embeddings = cmt_embeddings.to(self.device)
            if condition is not None:
                condition = condition.to(self.device)

            # Translate NL to CM
            translated_nlt = self.model.translate_nlt_to_cmt(
                nlt_embeddings, condition
            )

            # Translate CM to NL
            translated_cmt = self.model.translate_cmt_to_nlt(
                cmt_embeddings, condition
            )

            # Compute pairwise similarities
            nlt_sim = torch.mm(nlt_embeddings, nlt_embeddings.t())
            translated_nlt_sim = torch.mm(
                translated_nlt, translated_nlt.t())

            cmt_sim = torch.mm(cmt_embeddings, cmt_embeddings.t())
            translated_cmt_sim = torch.mm(
                translated_cmt, translated_cmt.t())

            # Compute correlations
            nlt_corr = torch.corrcoef(torch.stack([
                nlt_sim.flatten(), translated_nlt_sim.flatten()
            ]))[0, 1].item()

            cmt_corr = torch.corrcoef(torch.stack([
                cmt_sim.flatten(), translated_cmt_sim.flatten()
            ]))[0, 1].item()

            return {
                'nlt_geometry_correlation': nlt_corr,
                'cmt_geometry_correlation': cmt_corr,
                'mean_geometry_correlation': (nlt_corr + cmt_corr) / 2
            }

    def compute_clustering_metrics(
        self,
        embeddings: torch.Tensor,
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
        with torch.no_grad():
            embeddings = embeddings.to(self.device)
            embeddings_np = embeddings.cpu().numpy()

            if n_clusters is None:
                n_clusters = len(np.unique(labels))

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_np)

            # Compute metrics
            ari = adjusted_rand_score(labels, cluster_labels)
            nmi = normalized_mutual_info_score(labels, cluster_labels)

            return {
                'ari': ari,
                'nmi': nmi
            }

    def evaluate(
        self,
        nlt_embeddings: torch.Tensor,
        cmt_embeddings: torch.Tensor,
        nlt_labels: Optional[np.ndarray] = None,
        cmt_labels: Optional[np.ndarray] = None,
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.

        Args:
            nlt_embeddings: NL embeddings
            cmt_embeddings: CM embeddings
            nlt_labels: NL labels for clustering
            cmt_labels: CM labels for clustering
            condition: Optional conditioning vector

        Returns:
            Dictionary of all evaluation metrics
        """
        results = {}

        # Basic translation metrics
        results['cosine_similarity'] = self.compute_cosine_similarity(
            nlt_embeddings, cmt_embeddings
        )

        results['top_1_accuracy'] = self.compute_top_k_accuracy(
            nlt_embeddings, cmt_embeddings, k=1
        )

        results['top_5_accuracy'] = self.compute_top_k_accuracy(
            nlt_embeddings, cmt_embeddings, k=5
        )

        results['mean_rank'] = self.compute_mean_rank(
            nlt_embeddings, cmt_embeddings
        )

        results['mrr'] = self.compute_mrr(
            nlt_embeddings, cmt_embeddings
        )

        # Cycle consistency metrics
        cycle_metrics = self.compute_cycle_consistency_metrics(
            nlt_embeddings, cmt_embeddings, condition
        )
        results.update(cycle_metrics)

        # Geometry preservation metrics
        geometry_metrics = self.compute_geometry_preservation_metrics(
            nlt_embeddings, cmt_embeddings, condition
        )
        results.update(geometry_metrics)

        # Clustering metrics
        if nlt_labels is not None:
            nlt_clustering = self.compute_clustering_metrics(
                nlt_embeddings, nlt_labels
            )
            results.update(
                {f'nlt_{k}': v for k, v in nlt_clustering.items()})

        if cmt_labels is not None:
            cmt_clustering = self.compute_clustering_metrics(
                cmt_embeddings, cmt_labels
            )
            results.update(
                {f'cmt_{k}': v for k, v in cmt_clustering.items()})

        return results
    
    def evaluate_batch(self, batch: Dict[str, torch.Tensor], condition: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Evaluate a batch of embeddings.
        """
        return self.evaluate(batch['nlt'].to(self.device), batch['cmt'].to(self.device), condition=condition)

    def evaluate_loader(self, loader: DataLoader, condition: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Evaluate a loader of embeddings.
        """
        batch_results = {}
        for batch in tqdm(loader, desc="Evaluating Test Set"):
            result = self.evaluate_batch(batch, condition)
            for key, value in result.items():
                batch_results[key] = [value] if key not in batch_results else batch_results[key] + [value]
                
        for key, value in batch_results.items():
            batch_results[key] = np.mean(value)
            
        return batch_results

    def create_evaluation_table(
        self,
        results: Dict[str, Any],
        baseline_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a formatted evaluation table.

        Args:
            results: Evaluation results
            baseline_results: Optional baseline results for comparison

        Returns:
            Formatted table string
        """
        table_lines = []
        table_lines.append("=" * 80)
        table_lines.append("CMVec2Vec Translation Evaluation Results")
        table_lines.append("=" * 80)
        table_lines.append("")

        # Basic metrics
        table_lines.append("Basic Translation Metrics:")
        table_lines.append("-" * 40)
        table_lines.append(
            f"{'Cosine Similarity':<25} {results.get('cosine_similarity', 0):.4f}")
        table_lines.append(
            f"{'Top-1 Accuracy':<25} {results.get('top_1_accuracy', 0):.4f}")
        table_lines.append(
            f"{'Top-5 Accuracy':<25} {results.get('top_5_accuracy', 0):.4f}")
        table_lines.append(
            f"{'Mean Rank':<25} {results.get('mean_rank', 0):.2f}")
        table_lines.append(f"{'MRR':<25} {results.get('mrr', 0):.4f}")
        table_lines.append("")

        # Cycle consistency
        table_lines.append("Cycle Consistency:")
        table_lines.append("-" * 40)
        table_lines.append(
            f"{'NLT Cycle Sim':<25} {results.get('nlt_cycle_similarity', 0):.4f}")
        table_lines.append(
            f"{'CMT Cycle Sim':<25} {results.get('cmt_cycle_similarity', 0):.4f}")
        table_lines.append(
            f"{'Mean Cycle Sim':<25} {results.get('mean_cycle_similarity', 0):.4f}")
        table_lines.append("")

        # Geometry preservation
        table_lines.append("Geometry Preservation:")
        table_lines.append("-" * 40)
        table_lines.append(
            f"{'NLT Geometry Corr':<25} {results.get('nlt_geometry_correlation', 0):.4f}")
        table_lines.append(
            f"{'CMT Geometry Corr':<25} {results.get('cmt_geometry_correlation', 0):.4f}")
        table_lines.append(
            f"{'Mean Geometry Corr':<25} {results.get('mean_geometry_correlation', 0):.4f}")
        table_lines.append("")

        # Clustering metrics
        if 'nlt_ari' in results or 'cmt_ari' in results:
            table_lines.append("Clustering Metrics:")
            table_lines.append("-" * 40)
            if 'nlt_ari' in results:
                table_lines.append(
                    f"{'NLT ARI':<25} {results.get('nlt_ari', 0):.4f}")
                table_lines.append(
                    f"{'NLT NMI':<25} {results.get('nlt_nmi', 0):.4f}")
            if 'cmt_ari' in results:
                table_lines.append(
                    f"{'CMT ARI':<25} {results.get('cmt_ari', 0):.4f}")
                table_lines.append(
                    f"{'CMT NMI':<25} {results.get('cmt_nmi', 0):.4f}")
            table_lines.append("")

        return "\n".join(table_lines)

    def plot_embeddings(
        self,
        nlt_embeddings: torch.Tensor,
        cmt_embeddings: torch.Tensor,
        translated_nlt: Optional[torch.Tensor] = None,
        translated_cmt: Optional[torch.Tensor] = None,
        method: str = 'tsne',
        save_path: str = "logs/cm_vec2vec/"
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
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

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

            # Dimensionality reduction
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            else:
                raise ValueError(f"Unknown method: {method}")

            reduced_embeddings = reducer.fit_transform(all_embeddings)

            # Create plot
            plt.figure(figsize=(12, 8))

            # Plot by type
            unique_labels = list(set(all_labels))
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = np.array(all_labels) == label
                plt.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    c=[colors[i]],
                    label=label,
                    alpha=0.7,
                    s=50
                )

            plt.legend()
            plt.title(f"Embeddings Visualization ({method.upper()})")
            plt.xlabel(f"{method.upper()} 1")
            plt.ylabel(f"{method.upper()} 2")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()
