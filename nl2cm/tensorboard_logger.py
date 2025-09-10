"""
TensorBoard Logger for NL2CM

This module provides TensorBoard logging functionality for tracking training
losses, validation metrics, and evaluation results.
"""

import os
import torch
from tensorboardX import SummaryWriter
from typing import Dict, Optional, Any
import time


class NL2CMTensorBoardLogger:
    """
    TensorBoard logger for NL2CM training and evaluation.

    This logger tracks:
    - Training losses (generator, discriminator, individual components)
    - Validation losses
    - Evaluation metrics (cosine similarity, rank, accuracy, etc.)
    - Model parameters and gradients
    - Learning rate schedules
    """

    def __init__(self, log_dir: str, experiment_name: str = "nl2cm_experiment"):
        """
        Initialize the TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # Create log directory
        self.full_log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.full_log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.full_log_dir)

        # Track global step for training
        self.global_step = 0
        self.epoch = 0

        print(f"TensorBoard logging to: {self.full_log_dir}")
        print(f"To view logs, run: tensorboard --logdir {self.full_log_dir}")

    def log_training_losses(self, losses: Dict[str, float], epoch: int, step: int = None):
        """
        Log training losses to TensorBoard.

        Args:
            losses: Dictionary of loss values
            epoch: Current epoch
            step: Current step (if None, uses global step)
        """
        if step is None:
            step = self.global_step

        # Log individual loss components
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, (int, float)):
                self.writer.add_scalar(
                    f'Training/{loss_name}', loss_value, step)

        # Log total losses
        if 'total' in losses:
            self.writer.add_scalar(
                'Training/Total_Generator_Loss', losses['total'], step)
        if 'disc_total' in losses:
            self.writer.add_scalar(
                'Training/Total_Discriminator_Loss', losses['disc_total'], step)

        # Log loss ratios for monitoring training balance
        if 'total' in losses and 'disc_total' in losses:
            ratio = losses['total'] / (losses['disc_total'] + 1e-8)
            self.writer.add_scalar('Training/Gen_Disc_Ratio', ratio, step)

        self.writer.flush()

    def log_validation_losses(self, losses: Dict[str, float], epoch: int):
        """
        Log validation losses to TensorBoard.

        Args:
            losses: Dictionary of validation loss values
            epoch: Current epoch
        """
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, (int, float)):
                self.writer.add_scalar(
                    f'Validation/{loss_name}', loss_value, epoch)

        # Log total losses
        if 'total' in losses:
            self.writer.add_scalar(
                'Validation/Total_Generator_Loss', losses['total'], epoch)
        if 'disc_total' in losses:
            self.writer.add_scalar(
                'Validation/Total_Discriminator_Loss', losses['disc_total'], epoch)

        self.writer.flush()

    def log_evaluation_metrics(self, metrics: Dict[str, float], epoch: int = None, prefix: str = "Evaluation"):
        """
        Log evaluation metrics to TensorBoard.

        Args:
            metrics: Dictionary of evaluation metrics
            epoch: Current epoch (if None, uses global step)
            prefix: Prefix for metric names
        """
        step = epoch if epoch is not None else self.global_step

        # Basic translation metrics
        basic_metrics = ['cosine_similarity', 'mean_rank', 'top_1_accuracy',
                         'top_5_accuracy', 'top_10_accuracy', 'mrr']

        for metric in basic_metrics:
            if metric in metrics:
                self.writer.add_scalar(
                    f'{prefix}/Basic/{metric}', metrics[metric], step)

        # Cycle consistency metrics
        cycle_metrics = ['nlt_cycle_similarity',
                         'cmt_cycle_similarity', 'mean_cycle_similarity']
        for metric in cycle_metrics:
            if metric in metrics:
                self.writer.add_scalar(
                    f'{prefix}/Cycle_Consistency/{metric}', metrics[metric], step)

        # Geometry preservation metrics
        geometry_metrics = ['nlt_geometry_correlation',
                            'cmt_geometry_correlation', 'mean_geometry_correlation']
        for metric in geometry_metrics:
            if metric in metrics:
                self.writer.add_scalar(
                    f'{prefix}/Geometry_Preservation/{metric}', metrics[metric], step)

        # Classification metrics
        classification_metrics = ['translated_ari', 'translated_nmi', 'original_ari',
                                  'original_nmi', 'ari_improvement', 'nmi_improvement']
        for metric in classification_metrics:
            if metric in metrics:
                self.writer.add_scalar(
                    f'{prefix}/Classification/{metric}', metrics[metric], step)

        self.writer.flush()

    def log_learning_rates(self, lr_generator: float, lr_discriminator: float, epoch: int):
        """
        Log learning rates to TensorBoard.

        Args:
            lr_generator: Generator learning rate
            lr_discriminator: Discriminator learning rate
            epoch: Current epoch
        """
        self.writer.add_scalar('Learning_Rate/Generator', lr_generator, epoch)
        self.writer.add_scalar(
            'Learning_Rate/Discriminator', lr_discriminator, epoch)
        self.writer.flush()

    def log_model_parameters(self, model: torch.nn.Module, epoch: int):
        """
        Log model parameters and gradients to TensorBoard.

        Args:
            model: The model to log
            epoch: Current epoch
        """
        # Log parameter histograms
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    f'Parameters/{name}', param.data, epoch)
                self.writer.add_histogram(
                    f'Gradients/{name}', param.grad.data, epoch)

        # Log parameter norms
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        self.writer.add_scalar('Model/Gradient_Norm', total_norm, epoch)
        self.writer.flush()

    def log_embeddings(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None,
                       epoch: int = None, tag: str = "embeddings"):
        """
        Log embeddings to TensorBoard for visualization.

        Args:
            embeddings: Embeddings tensor (N, D)
            labels: Optional labels for coloring (N,)
            epoch: Current epoch
            tag: Tag for the embeddings
        """
        step = epoch if epoch is not None else self.global_step

        # Use PCA to reduce dimensionality for visualization
        try:
            from sklearn.decomposition import PCA

            # Convert to numpy
            emb_np = embeddings.detach().cpu().numpy()

            # Apply PCA to reduce to 3D
            if emb_np.shape[1] > 3:
                pca = PCA(n_components=3)
                emb_reduced = pca.fit_transform(emb_np)
            else:
                emb_reduced = emb_np

            # Convert back to tensor
            emb_tensor = torch.FloatTensor(emb_reduced)

            # Add metadata if labels provided
            metadata = None
            if labels is not None:
                metadata = labels.detach().cpu().numpy().tolist()

            self.writer.add_embedding(
                emb_tensor,
                metadata=metadata,
                tag=tag,
                global_step=step
            )

        except ImportError:
            print("Warning: sklearn not available for embedding visualization")

        self.writer.flush()

    def log_translation_examples(self, nlt_emb: torch.Tensor, cmt_emb: torch.Tensor,
                                 translated_emb: torch.Tensor, epoch: int = None,
                                 n_examples: int = 10):
        """
        Log translation examples for visualization.

        Args:
            nlt_emb: Original NL embeddings
            cmt_emb: Target CM embeddings
            translated_emb: Translated embeddings
            epoch: Current epoch
            n_examples: Number of examples to log
        """
        step = epoch if epoch is not None else self.global_step

        # Sample random examples
        indices = torch.randperm(len(nlt_emb))[:n_examples]

        # Compute similarities
        similarities_orig = torch.nn.functional.cosine_similarity(
            nlt_emb[indices], cmt_emb[indices], dim=1
        )
        similarities_trans = torch.nn.functional.cosine_similarity(
            translated_emb[indices], cmt_emb[indices], dim=1
        )

        # Log similarity improvements
        improvements = similarities_trans - similarities_orig
        self.writer.add_scalar('Translation/Mean_Similarity_Improvement',
                               improvements.mean().item(), step)
        self.writer.add_scalar('Translation/Original_Similarity',
                               similarities_orig.mean().item(), step)
        self.writer.add_scalar('Translation/Translated_Similarity',
                               similarities_trans.mean().item(), step)

        self.writer.flush()

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        Log hyperparameters and final metrics.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        # Convert all values to strings for TensorBoard
        hparams_str = {k: str(v) for k, v in hparams.items()}
        metrics_float = {
            k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

        self.writer.add_hparams(hparams_str, metrics_float)
        self.writer.flush()

    def log_text(self, text: str, tag: str = "text", step: int = None):
        """
        Log text to TensorBoard.

        Args:
            text: Text to log
            tag: Tag for the text
            step: Step number
        """
        if step is None:
            step = self.global_step

        self.writer.add_text(tag, text, step)
        self.writer.flush()

    def log_figure(self, figure, tag: str = "figure", step: int = None):
        """
        Log a matplotlib figure to TensorBoard.

        Args:
            figure: Matplotlib figure
            tag: Tag for the figure
            step: Step number
        """
        if step is None:
            step = self.global_step

        self.writer.add_figure(tag, figure, step)
        self.writer.flush()

    def increment_global_step(self):
        """Increment the global step counter."""
        self.global_step += 1

    def set_epoch(self, epoch: int):
        """Set the current epoch."""
        self.epoch = epoch

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.full_log_dir}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_tensorboard_logger(log_dir: str, experiment_name: str = None) -> NL2CMTensorBoardLogger:
    """
    Create a TensorBoard logger with timestamp.

    Args:
        log_dir: Base directory for logs
        experiment_name: Name of the experiment (if None, uses timestamp)

    Returns:
        NL2CMTensorBoardLogger instance
    """
    if experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"nl2cm_{timestamp}"

    return NL2CMTensorBoardLogger(log_dir, experiment_name)
