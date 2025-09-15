"""
Training utilities for CMVec2Vec
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from cm_vec2vec.evaluation import CMVec2VecEvaluator
from cm_vec2vec.translators.cm_vec2vec_translator import CMVec2VecTranslator
from cm_vec2vec.losses_cosine_distance import (
    compute_all_losses_with_correspondence,
    discriminator_loss,
)
from cm_vec2vec.utils import get_device


class CMVec2VecTrainer:
    """
    Trainer class for CMVec2Vec model.

    Handles training with all loss functions including adversarial, reconstruction,
    cycle consistency, and vector space preservation losses.
    """

    def __init__(
        self,
        model: CMVec2VecTranslator,
        lr_generator: float = 1e-4,
        lr_discriminator: float = 4e-4,
        loss_weights: Optional[Dict[str, float]] = None,
        gan_type: str = 'least_squares',
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        use_scheduler: bool = True,
        warmup_steps: int = 1000,
        save_dir: str = 'logs/cm_vec2vec',
        evaluator: Optional[CMVec2VecEvaluator] = None,
        # Enhanced loss parameters
        vsp_temperature: float = 1.0,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        cycle_margin: float = 0.1,
        target_losses: Optional[Dict[str, float]] = None,
        enhanced: bool = False
    ):
        """
        Initialize the trainer.

        Args:
            model: CMVec2Vec model to train
            lr_generator: Learning rate for generator
            lr_discriminator: Learning rate for discriminators
            loss_weights: Weights for different loss components
            gan_type: Type of GAN loss
            weight_decay: Weight decay for optimizers
            max_grad_norm: Maximum gradient norm for clipping
            use_scheduler: Whether to use learning rate scheduler
            warmup_steps: Number of warmup steps for scheduler
            save_dir: Directory to save checkpoints and logs
            evaluator: Optional evaluator for validation metrics
            vsp_temperature: Temperature for enhanced VSP loss
            focal_alpha: Alpha parameter for focal adversarial loss
            focal_gamma: Gamma parameter for focal adversarial loss
            cycle_margin: Margin for cycle consistency loss
            target_losses: Target loss values for adaptive weighting
            enhanced: Whether to use enhanced loss functions
        """
        self.model = model
        self.device = get_device()
        self.gan_type = gan_type
        self.max_grad_norm = max_grad_norm
        self.use_scheduler = use_scheduler
        self.warmup_steps = warmup_steps
        self.evaluator = evaluator
        # Enhanced loss parameters
        self.vsp_temperature = vsp_temperature
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cycle_margin = cycle_margin
        self.target_losses = target_losses
        self.enhanced = enhanced
        # Use provided loss weights or defaults (matching parse_args.py defaults)
        if loss_weights is not None:
            self.loss_weights = loss_weights
        else:
            self.loss_weights = {
                'reconstruction': 10.0,
                'cycle_consistency': 10.0,
                'vsp': 5.0,
                'adversarial': 2.0,
                'latent_adversarial': 2.0,
                'correspondence': 20.0,
                'cosine_correspondence': 5.0,
                'ranking': 15.0
            }

        # Move model to device
        self.model.to(self.device)

        # Setup optimizers
        self._setup_optimizers(lr_generator, lr_discriminator, weight_decay)

        # Setup schedulers
        if use_scheduler:
            self._setup_schedulers()

        # Training history
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'epoch_losses': []
        }

        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir=save_dir)

    def _setup_optimizers(self, lr_generator: float, lr_discriminator: float, weight_decay: float):
        """Setup optimizers for generator and discriminators."""
        # Generator optimizer (translator)
        generator_params = []
        for name, param in self.model.named_parameters():
            if 'discriminator' not in name:
                generator_params.append(param)

        self.optimizer_generator = optim.Adam(
            generator_params,
            lr=lr_generator,
            weight_decay=weight_decay,
            betas=(0.5, 0.999)
        )

        # Discriminator optimizers
        self.optimizer_discriminators = {}
        discriminator_names = ['nlt_discriminator',
                               'cmt_discriminator', 'latent_discriminator']
        for name in discriminator_names:
            discriminator: nn.Module = getattr(self.model, name)
            self.optimizer_discriminators[name] = optim.Adam(
                discriminator.parameters(),
                lr=lr_discriminator,
                weight_decay=weight_decay,
                betas=(0.5, 0.999)
            )

    def _setup_schedulers(self):
        """Setup learning rate schedulers."""
        # Generator scheduler
        self.scheduler_generator = optim.lr_scheduler.LambdaLR(
            self.optimizer_generator,
            lr_lambda=lambda step: min(1.0, step / self.warmup_steps)
        )

        # Discriminator schedulers
        self.scheduler_discriminators = {}
        for name, optimizer in self.optimizer_discriminators.items():
            self.scheduler_discriminators[name] = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(1.0, step / self.warmup_steps)
            )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Batch of embeddings with 'nlt' and 'cmt' keys
            condition: Optional conditioning vector

        Returns:
            Dictionary of losses for this step
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if condition is not None:
            condition = condition.to(self.device)

        # Forward pass
        reconstructions, translations, latent_reps = self.model(
            batch, condition=condition, include_reps=True, noise_level=0.01
        )

        # Get discriminator scores for real embeddings
        real_discriminator_scores = self.model.get_discriminator_scores(
            batch, condition)

        # Get discriminator scores for translated embeddings
        fake_discriminator_scores = self.model.get_discriminator_scores_for_translations(
            translations, condition)

        # Combine real and fake scores
        discriminator_scores = {
            **real_discriminator_scores, **fake_discriminator_scores}

        latent_discriminator_scores = self.model.get_latent_discriminator_scores(
            latent_reps, condition)

        # Compute generator losses
        gen_losses = compute_all_losses_with_correspondence(
            (reconstructions, translations),
            batch,
            discriminator_scores,
            latent_discriminator_scores,
            self.loss_weights,
            self.gan_type,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            margin=self.cycle_margin,
            temperature=self.vsp_temperature,
            enhanced=self.enhanced,
        )

        # Generator update
        self.optimizer_generator.zero_grad()
        gen_losses['total'].backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in self.model.named_parameters() if 'discriminator' not in n],
            self.max_grad_norm
        )
        self.optimizer_generator.step()

        # Discriminator updates
        disc_losses = {}
        for name, _ in self.optimizer_discriminators.items():
            if name in real_discriminator_scores:
                real_scores = real_discriminator_scores[name]
                fake_scores = fake_discriminator_scores.get(
                    f'{name}_fake', real_scores)

                disc_loss = discriminator_loss(
                    real_scores, fake_scores, self.gan_type)
                disc_losses[f'disc_{name}'] = disc_loss.item()

                self.optimizer_discriminators[name].zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    getattr(self.model, name).parameters(), self.max_grad_norm)
                self.optimizer_discriminators[name].step()

        # Update schedulers
        if self.use_scheduler:
            self.scheduler_generator.step()
            for scheduler in self.scheduler_discriminators.values():
                scheduler.step()

        # Convert losses to float for logging
        step_losses = {k: v.item() if torch.is_tensor(v)
                       else v for k, v in gen_losses.items()}
        step_losses.update(disc_losses)

        return step_losses

    def validate(
        self,
        val_loader: DataLoader,
        condition: Optional[torch.Tensor] = None,
        save_table: bool = False,
        compute_metrics: bool = False
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            condition: Optional conditioning vector
            save_table: Whether to save evaluation table
            compute_metrics: Whether to compute expensive evaluation metrics

        Returns:
            Dictionary of validation losses
        """
        self.model.eval()
        total_scores = {}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Computing validation losses{' and metrics' if compute_metrics else ''}"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if condition is not None:
                    condition = condition.to(self.device)

                # Forward pass
                reconstructions, translations, latent_reps = self.model(
                    batch, condition=condition, include_reps=True
                )

                # Get discriminator scores for real embeddings
                real_discriminator_scores = self.model.get_discriminator_scores(
                    batch, condition)

                # Get discriminator scores for translated embeddings
                fake_discriminator_scores = self.model.get_discriminator_scores_for_translations(
                    translations, condition)

                # Combine real and fake scores
                discriminator_scores = {
                    **real_discriminator_scores, **fake_discriminator_scores}

                latent_discriminator_scores = self.model.get_latent_discriminator_scores(
                    latent_reps, condition)

                # Compute losses
                losses = compute_all_losses_with_correspondence(
                    (reconstructions, translations),
                    batch,
                    discriminator_scores,
                    latent_discriminator_scores,
                    self.loss_weights,
                    self.gan_type,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    margin=self.cycle_margin,
                    temperature=self.vsp_temperature,
                    enhanced=self.enhanced
                )

                # Accumulate losses
                for key, value in losses.items():
                    if key not in total_scores:
                        total_scores[key] = 0.0
                    total_scores[key] += value.item() if torch.is_tensor(value) else value

                num_batches += 1

        # Average losses
        avg_scores = {k: v / num_batches for k, v in total_scores.items()}

        if self.evaluator is not None and compute_metrics:
            avg_scores.update(self.evaluator.evaluate_loader(
                val_loader, condition=condition, save_table=save_table))

        return avg_scores

    def validate_lightweight(
        self,
        val_loader: DataLoader,
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Lightweight validation that only computes basic losses without expensive metrics.

        Args:
            val_loader: Validation data loader
            condition: Optional conditioning vector

        Returns:
            Dictionary of validation losses
        """
        self.model.eval()
        total_scores = {}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Computing lightweight validation losses"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if condition is not None:
                    condition = condition.to(self.device)

                # Forward pass
                reconstructions, translations, latent_reps = self.model(
                    batch, condition=condition, include_reps=True
                )

                # Get discriminator scores for real embeddings
                real_discriminator_scores = self.model.get_discriminator_scores(
                    batch, condition)

                # Get discriminator scores for translated embeddings
                fake_discriminator_scores = self.model.get_discriminator_scores_for_translations(
                    translations, condition)

                # Combine real and fake scores
                discriminator_scores = {
                    **real_discriminator_scores, **fake_discriminator_scores}

                latent_discriminator_scores = self.model.get_latent_discriminator_scores(
                    latent_reps, condition)

                # Compute losses
                losses = compute_all_losses_with_correspondence(
                    (reconstructions, translations),
                    batch,
                    discriminator_scores,
                    latent_discriminator_scores,
                    self.loss_weights,
                    self.gan_type,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    margin=self.cycle_margin,
                    temperature=self.vsp_temperature,
                    enhanced=self.enhanced
                )

                # Accumulate losses
                for key, value in losses.items():
                    if key not in total_scores:
                        total_scores[key] = 0.0
                    total_scores[key] += value.item() if torch.is_tensor(value) else value

                num_batches += 1

        # Average losses
        avg_scores = {k: v / num_batches for k, v in total_scores.items()}
        return avg_scores

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_every: int = 10,
        early_stopping_patience: int = 20,
        condition: Optional[torch.Tensor] = None,
        save_table: bool = False,
        val_frequency: int = 20,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience
            condition: Optional conditioning vector
            save_table: Whether to save evaluation table
            val_frequency: Run validation every N epochs (default: 1)

        Returns:
            Training history
        """
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        checkpoints_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            epoch_losses = []
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in train_pbar:
                step_losses = self.train_step(batch, condition)
                epoch_losses.append(step_losses)

                # Update progress bar
                train_pbar.set_postfix({
                    'total_loss': step_losses.get('total', 0),
                    'rec_loss': step_losses.get('reconstruction', 0),
                    'adv_loss': step_losses.get('adversarial', 0),
                    'vsp_loss': step_losses.get('vsp', 0)
                })

            # Average epoch losses
            avg_epoch_losses = {}
            for key in epoch_losses[0].keys():
                avg_epoch_losses[key] = np.mean(
                    [loss[key] for loss in epoch_losses])

            for key, value in avg_epoch_losses.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)

            self.history['train_losses'].append(avg_epoch_losses)
            self.history['epoch_losses'].append(avg_epoch_losses)

            # train_scores = self.evaluator.evaluate_loader(
            #     train_loader, condition, save_table=save_table)
            # for key, value in train_scores.items():
            #     self.writer.add_scalar(f'train/{key}', value, epoch)

            # Validation
            if val_loader is not None:
                # compute_val_metrics = (epoch + 1) % val_frequency == 0
                import time
                start_time = time.time()
                val_scores = self.validate(
                    val_loader, condition, 
                    save_table=save_table, 
                    compute_metrics=False
                )
                print(
                    f"Validation time: {time.time() - start_time:.2f} seconds")
                for key, value in val_scores.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                self.history['val_losses'].append(val_scores)

                # Early stopping
                val_total_loss = val_scores.get('total', float('inf'))
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint(
                        os.path.join(checkpoints_dir, 'best_model.pt'))
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                print(f"Epoch {epoch+1}: Train Loss = {avg_epoch_losses.get('total', 0):.4f}, "
                      f"Val Loss = {val_total_loss:.4f}")
            else:
                print(
                    f"Epoch {epoch+1}: Train Loss = {avg_epoch_losses.get('total', 0):.4f}")

            # Save checkpoint
            # if (epoch + 1) % save_every == 0:
            #     self.save_checkpoint(os.path.join(
            #         checkpoints_dir, f'epoch_{epoch+1}.pt'))

        # Save final model
        # self.save_checkpoint(os.path.join(checkpoints_dir, 'final_model.pt'))

        return self.history

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
            'optimizer_discriminators_state_dict': {
                name: opt.state_dict() for name, opt in self.optimizer_discriminators.items()
            },
            'loss_weights': self.loss_weights,
            'history': self.history
        }

        if self.use_scheduler:
            checkpoint['scheduler_generator_state_dict'] = self.scheduler_generator.state_dict()
            checkpoint['scheduler_discriminators_state_dict'] = {
                name: sched.state_dict() for name, sched in self.scheduler_discriminators.items()
            }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_generator.load_state_dict(
            checkpoint['optimizer_generator_state_dict'])

        for name, opt_state in checkpoint['optimizer_discriminators_state_dict'].items():
            if name in self.optimizer_discriminators:
                self.optimizer_discriminators[name].load_state_dict(opt_state)

        if 'scheduler_generator_state_dict' in checkpoint and self.use_scheduler:
            self.scheduler_generator.load_state_dict(
                checkpoint['scheduler_generator_state_dict'])

        if 'scheduler_discriminators_state_dict' in checkpoint and self.use_scheduler:
            for name, sched_state in checkpoint['scheduler_discriminators_state_dict'].items():
                if name in self.scheduler_discriminators:
                    self.scheduler_discriminators[name].load_state_dict(
                        sched_state)

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f"Checkpoint loaded from {filepath}")

    def translate_nlt_to_cmt(
        self,
        nlt_embeddings: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Translate NL embeddings to CM embeddings.

        Args:
            nlt_embeddings: NL embeddings
            condition: Optional conditioning vector

        Returns:
            Translated CM embeddings
        """
        self.model.eval()
        with torch.no_grad():
            nlt_embeddings = nlt_embeddings.to(self.device)
            if condition is not None:
                condition = condition.to(self.device)

            return self.model.translate_nlt_to_cmt(nlt_embeddings, condition)

    def translate_cmt_to_nlt(
        self,
        cmt_embeddings: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Translate CM embeddings to NL embeddings.

        Args:
            cmt_embeddings: CM embeddings
            condition: Optional conditioning vector

        Returns:
            Translated NL embeddings
        """
        self.model.eval()
        with torch.no_grad():
            cmt_embeddings = cmt_embeddings.to(self.device)
            if condition is not None:
                condition = condition.to(self.device)

            return self.model.translate_cmt_to_nlt(cmt_embeddings, condition)
