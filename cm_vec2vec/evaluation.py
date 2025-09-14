"""
Evaluation utilities for CMVec2Vec
"""

import torch
import numpy as np
from typing import Callable, Dict, Optional, Any, Union
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from cm_vec2vec.translators.cm_vec2vec_translator import CMVec2VecTranslator
from cm_vec2vec.utils import (
    get_device,
    plot_embeddings,
    create_evaluation_table,
    compute_baseline_metrics,
    compute_retrieval_metrics,
    compute_clustering_metrics,
    compute_cosine_similarity,
)


class CMVec2VecEvaluator:
    """
    Evaluator class for CMVec2Vec model.

    Provides comprehensive evaluation metrics including translation quality,
    cycle consistency, geometry preservation, and clustering metrics.
    """

    def __init__(
        self, 
        model: CMVec2VecTranslator, 
        save_dir: str = 'logs/cm_vec2vec',        
    ):
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

    def _get_translated_batch_embeddings(self, embeddings: torch.Tensor, condition: Optional[torch.Tensor] = None, transition: str = None) -> torch.Tensor:
        embeddings = embeddings.to(self.device)
        if condition is not None:
            condition = condition.to(self.device)
        if transition == 'nlt2cmt':
            return self.model.translate_nlt_to_cmt(embeddings, condition)
        elif transition == 'cmt2nlt':
            return self.model.translate_cmt_to_nlt(embeddings, condition)
        else:
            raise ValueError(f"Invalid transition: {transition}")

    def get_nlt_to_cmt_embeddings(self, batch: Union[torch.Tensor, DataLoader], condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings from a batch.
        """ 
        return self._get_translated_batch_embeddings(batch, condition, transition='nlt2cmt')

    def get_cmt_to_nlt_embeddings(self, batch: Union[torch.Tensor, DataLoader], condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings from a batch.    
        """
        return self._get_translated_batch_embeddings(batch, condition, transition='cmt2nlt')


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
            nlt_cycle_sim = compute_cosine_similarity(
                nlt_embeddings, cycle_nlt)
            cmt_cycle_sim = compute_cosine_similarity(
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

            def safe_correlation(x, y):
                """Compute correlation with NaN handling."""
                try:
                    # Check for constant values (no variance)
                    if torch.std(x) < 1e-8 or torch.std(y) < 1e-8:
                        return 0.0

                    # Compute correlation
                    corr = torch.corrcoef(torch.stack([x, y]))[0, 1]

                    # Handle NaN values
                    if torch.isnan(corr):
                        return 0.0

                    return corr.item()
                except Exception:
                    return 0.0

            # Compute correlations with error handling
            nlt_corr = safe_correlation(
                nlt_sim.flatten(), translated_nlt_sim.flatten()
            )

            cmt_corr = safe_correlation(
                cmt_sim.flatten(), translated_cmt_sim.flatten()
            )

            return {
                'nlt_geometry_correlation': nlt_corr,
                'cmt_geometry_correlation': cmt_corr,
                'mean_geometry_correlation': (nlt_corr + cmt_corr) / 2
            }
    
    def evaluate(
        self, 
        nlt_embeddings: torch.Tensor, 
        cmt_embeddings: torch.Tensor, 
        nlt_labels: Optional[torch.Tensor] = None, 
        cmt_labels: Optional[torch.Tensor] = None, 
        condition: Optional[torch.Tensor] = None, 
        plot: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate embeddings.
        """
        all_results = {}
        
        nlt_to_cmt_embeddings = self.get_nlt_to_cmt_embeddings(nlt_embeddings, condition=condition)
        cmt_to_nlt_embeddings = self.get_cmt_to_nlt_embeddings(cmt_embeddings, condition=condition)
        
        nlt_to_cmt_translation_results = compute_retrieval_metrics(nlt_to_cmt_embeddings, cmt_embeddings)
        cmt_to_nlt_translation_results = compute_retrieval_metrics(cmt_to_nlt_embeddings, nlt_embeddings)
        
        retrieval_results = dict()
        for k, v in nlt_to_cmt_translation_results.items():
            retrieval_results[f'nlt2cmt_{k}'] = v
        for k, v in cmt_to_nlt_translation_results.items():
            retrieval_results[f'cmt2nlt_{k}'] = v
        
        cycle_consistency_results = self.compute_cycle_consistency_metrics(nlt_embeddings, cmt_embeddings, condition=condition)
        
        geometry_preservation_results = self.compute_geometry_preservation_metrics(nlt_embeddings, cmt_embeddings, condition=condition)
        
        
        nlt_clustering_results = compute_clustering_metrics(
            nlt_embeddings, nlt_labels, condition=condition
        ) if nlt_labels and nlt_labels[0] is not None else None
        cmt_clustering_results = compute_clustering_metrics(
            cmt_embeddings, cmt_labels, condition=condition
        ) if cmt_labels and cmt_labels[0] is not None else None
        
        clustering_results = {}
        if nlt_clustering_results is not None:
            for k, v in nlt_clustering_results.items():
                clustering_results[f'nlt_clustering_{k}'] = v
        if cmt_clustering_results is not None:
            for k, v in cmt_clustering_results.items():
                clustering_results[f'cmt_clustering_{k}'] = v
        
        baseline_nlt_to_cmt_results = compute_baseline_metrics(nlt_to_cmt_embeddings, cmt_embeddings)
        baseline_cmt_to_nlt_results = compute_baseline_metrics(cmt_to_nlt_embeddings, nlt_embeddings)
        
        baseline_results = {}
        for k, v in baseline_nlt_to_cmt_results.items():
            baseline_results[f'baseline_nlt_to_cmt_{k}'] = v
        for k, v in baseline_cmt_to_nlt_results.items():
            baseline_results[f'baseline_cmt_to_nlt_{k}'] = v
        
        if plot:
            plot_embeddings(
                nlt_embeddings, 
                cmt_embeddings, 
                nlt_to_cmt_embeddings, 
                cmt_to_nlt_embeddings, 
                condition=condition,
                save_path=self.save_dir
            )
        
        result_types = {
            'cycle_consistency_results': cycle_consistency_results,
            'geometry_preservation_results': geometry_preservation_results,
            'baseline_results': baseline_results,
            'clustering_results': clustering_results,
            'retrieval_results': retrieval_results
        }

        for result in result_types.values():
            for k, v in result.items():
                all_results[k] = v
        return all_results


    def evaluate_loader(
        self, 
        loader: DataLoader, 
        condition: Optional[torch.Tensor] = None, 
        plot: bool = False,
        save_table: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a loader of embeddings.
        """

        def get_labels(loader: DataLoader, key: str) -> Optional[torch.Tensor]:
            labels = [batch[key] if key in batch else None for batch in loader]
            if labels and labels[0] is not None:
                return torch.cat([torch.tensor(label) for label in labels], dim=0).to(self.device)
            return None
        
        nlt_embeddings = torch.cat([batch['nlt'] for batch in loader], dim=0).to(self.device)
        nlt_labels = get_labels(loader, 'nlt_label')
        cmt_embeddings = torch.cat([batch['cmt'] for batch in loader], dim=0).to(self.device)
        cmt_labels = get_labels(loader, 'cmt_label')
        
        results = self.evaluate(nlt_embeddings, cmt_embeddings, nlt_labels, cmt_labels, condition, plot)
        if save_table:
            print("Evaluation Table:")
            print(create_evaluation_table(results))
        return results
