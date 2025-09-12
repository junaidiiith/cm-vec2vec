"""
NL2CM Training Module

This module implements the training loop and loss functions for the NL2CM translation model,
following the vec2vec approach with adversarial, reconstruction, cycle consistency, and VSP losses.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np
from tqdm.auto import tqdm
import os

from nl2cm.utils import get_device
from nl2cm.tensorboard_logger import NL2CMTensorBoardLogger


class NL2CMTrainer:
    """
    Trainer for the NL2CM translation model.

    This trainer implements the complete training procedure including all loss functions
    from the vec2vec approach: adversarial, reconstruction, cycle consistency, and VSP.
    """

    def __init__(self, model: nn.Module,
        lr_generator: float = 1e-4, lr_discriminator: float = 4e-4,
        lambda_rec: float = 15.0, lambda_cyc: float = 15.0,
        lambda_vsp: float = 2.0, lambda_adv: float = 1.0,
        lambda_latent: float = 1.0, weight_decay: float = 0.01,
        use_tensorboard: bool = True, log_dir: str = 'tensorboard_logs'
    ):
        """
        Initialize the trainer.

        Args:
            model: The NL2CM translation model
            lr_generator: Learning rate for generator components
            lr_discriminator: Learning rate for discriminators
            lambda_rec: Weight for reconstruction loss
            lambda_cyc: Weight for cycle consistency loss
            lambda_vsp: Weight for vector space preservation loss
            lambda_adv: Weight for adversarial loss
            lambda_latent: Weight for latent adversarial loss
            weight_decay: Weight decay for optimizers
            use_tensorboard: Whether to use TensorBoard logging
            log_dir: Directory for TensorBoard logs
        """
        self.device = get_device()
        self.model = model.to(self.device)

        # Loss weights
        self.lambda_rec = lambda_rec
        self.lambda_cyc = lambda_cyc
        self.lambda_vsp = lambda_vsp
        self.lambda_adv = lambda_adv
        self.lambda_latent = lambda_latent

        # Setup optimizers
        self._setup_optimizers(lr_generator, lr_discriminator, weight_decay)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0  # Initialize epoch counter

        # TensorBoard logging
        self.use_tensorboard = use_tensorboard
        self.tensorboard_logger = None
        if use_tensorboard:
            self.tensorboard_logger = NL2CMTensorBoardLogger(log_dir)


    def _setup_optimizers(self, lr_generator: float, lr_discriminator: float,
                          weight_decay: float):
        """Setup optimizers for different components."""
        # Generator optimizer (adapters and backbone)
        generator_params = list(self.model.nlt_adapter.parameters()) + \
            list(self.model.cmt_adapter.parameters()) + \
            list(self.model.backbone.parameters()) + \
            list(self.model.nlt_output_adapter.parameters()) + \
            list(self.model.cmt_output_adapter.parameters())

        self.optimizer_generator = optim.AdamW(
            generator_params, lr=lr_generator, weight_decay=weight_decay
        )

        # Discriminator optimizer
        discriminator_params = list(self.model.nlt_discriminator.parameters()) + \
            list(self.model.cmt_discriminator.parameters()) + \
            list(self.model.latent_discriminator.parameters())

        self.optimizer_discriminator = optim.AdamW(
            discriminator_params, lr=lr_discriminator, weight_decay=weight_decay
        )

    def adversarial_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss using least squares GAN."""
        real_loss = F.mse_loss(real, torch.ones_like(real))
        fake_loss = F.mse_loss(fake, torch.zeros_like(fake))
        return (real_loss + fake_loss) / 2

    def generator_adversarial_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """Compute generator adversarial loss."""
        return F.mse_loss(fake, torch.ones_like(fake))

    def reconstruction_loss(self, outputs: Dict[str, torch.Tensor],
                            batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute reconstruction loss."""
        nlt_recon_loss = self.mse_loss(outputs['nlt_recon'], batch['nlt'])
        cmt_recon_loss = self.mse_loss(outputs['cmt_recon'], batch['cmt'])
        return nlt_recon_loss + cmt_recon_loss

    def cycle_consistency_loss(self, outputs: Dict[str, torch.Tensor],
                               batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cycle consistency loss."""
        # Forward cycle: nlt -> cmt -> nlt
        nlt_cycle = self.model.translate_cmt_to_nlt(outputs['nlt_to_cmt'])
        nlt_cyc_loss = self.mse_loss(nlt_cycle, batch['nlt'])

        # Backward cycle: cmt -> nlt -> cmt
        cmt_cycle = self.model.translate_nlt_to_cmt(outputs['cmt_to_nlt'])
        cmt_cyc_loss = self.mse_loss(cmt_cycle, batch['cmt'])

        return nlt_cyc_loss + cmt_cyc_loss

    def vector_space_preservation_loss(self, outputs: Dict[str, torch.Tensor],
                                       batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute vector space preservation loss."""
        nlt_emb = batch['nlt']
        cmt_emb = batch['cmt']
        nlt_to_cmt = outputs['nlt_to_cmt']
        cmt_to_nlt = outputs['cmt_to_nlt']

        # Compute pairwise dot products
        nlt_dots_orig = torch.mm(nlt_emb, nlt_emb.t())
        nlt_dots_trans = torch.mm(nlt_to_cmt, nlt_to_cmt.t())
        nlt_vsp_loss = self.mse_loss(nlt_dots_orig, nlt_dots_trans)

        cmt_dots_orig = torch.mm(cmt_emb, cmt_emb.t())
        cmt_dots_trans = torch.mm(cmt_to_nlt, cmt_to_nlt.t())
        cmt_vsp_loss = self.mse_loss(cmt_dots_orig, cmt_dots_trans)

        return nlt_vsp_loss + cmt_vsp_loss

    def compute_generator_loss(self, outputs: Dict[str, torch.Tensor],
                               batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all generator losses."""
        # Get discriminator outputs
        disc_outputs = self.model.get_discriminator_outputs(batch, outputs)

        # Adversarial losses
        nlt_adv_loss = self.generator_adversarial_loss(
            disc_outputs['nlt_fake'])
        cmt_adv_loss = self.generator_adversarial_loss(
            disc_outputs['cmt_fake'])
        latent_adv_loss = self.generator_adversarial_loss(
            disc_outputs['latent_fake'])

        # Other losses
        rec_loss = self.reconstruction_loss(outputs, batch)
        cyc_loss = self.cycle_consistency_loss(outputs, batch)
        vsp_loss = self.vector_space_preservation_loss(outputs, batch)

        # Total generator loss
        total_loss = (self.lambda_adv * (nlt_adv_loss + cmt_adv_loss) +
                      self.lambda_latent * latent_adv_loss +
                      self.lambda_rec * rec_loss +
                      self.lambda_cyc * cyc_loss +
                      self.lambda_vsp * vsp_loss)

        return {
            'total': total_loss,
            'adversarial': nlt_adv_loss + cmt_adv_loss,
            'latent_adversarial': latent_adv_loss,
            'reconstruction': rec_loss,
            'cycle_consistency': cyc_loss,
            'vector_space_preservation': vsp_loss
        }

    def compute_discriminator_loss(self, outputs: Dict[str, torch.Tensor],
                                   batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute discriminator losses."""
        disc_outputs = self.model.get_discriminator_outputs(batch, outputs)

        # Output space discriminator losses
        nlt_disc_loss = self.adversarial_loss(
            disc_outputs['nlt_real'], disc_outputs['nlt_fake'])
        cmt_disc_loss = self.adversarial_loss(
            disc_outputs['cmt_real'], disc_outputs['cmt_fake'])

        # Latent space discriminator loss
        latent_disc_loss = self.adversarial_loss(
            disc_outputs['latent_real'], disc_outputs['latent_fake'])

        total_loss = nlt_disc_loss + cmt_disc_loss + latent_disc_loss

        return {
            'total': total_loss,
            'nlt': nlt_disc_loss,
            'cmt': cmt_disc_loss,
            'latent': latent_disc_loss
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(batch)

        # Compute generator losses
        gen_losses = self.compute_generator_loss(outputs, batch)

        # Update generator
        self.optimizer_generator.zero_grad()
        gen_losses['total'].backward()
        self.optimizer_generator.step()

        # Forward pass again for discriminator (to avoid in-place operations)
        outputs_disc = self.model(batch)

        # Compute discriminator losses
        disc_losses = self.compute_discriminator_loss(outputs_disc, batch)

        # Update discriminator
        self.optimizer_discriminator.zero_grad()
        disc_losses['total'].backward()
        self.optimizer_discriminator.step()

        # Combine losses for logging
        losses = {**gen_losses, **
                  {f'disc_{k}': v for k, v in disc_losses.items()}}

        # Log to TensorBoard
        if self.tensorboard_logger:
            self.tensorboard_logger.log_training_losses(losses, self.epoch)
            self.tensorboard_logger.increment_global_step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}


    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_losses = dict()

        with torch.no_grad():
            for batch in val_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    batch = {k: v.to(get_device()) for k, v in batch.items()}
                else:
                    # Assume it's a tuple/list from TensorDataset
                    batch = {
                        'nlt': batch[0].to(get_device()), 
                        'cmt': batch[1].to(get_device())
                    }

                outputs = self.model(batch)

                gen_losses = self.compute_generator_loss(outputs, batch)
                disc_losses = self.compute_discriminator_loss(outputs, batch)

                # Accumulate losses
                for k, v in gen_losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v.item()
                for k, v in disc_losses.items():
                    total_losses[f'disc_{k}'] = total_losses.get(
                        f'disc_{k}', 0) + v.item()

        # Average losses
        num_batches = len(val_loader)
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        # Log to TensorBoard
        if self.tensorboard_logger:
            self.tensorboard_logger.log_validation_losses(avg_losses, epoch)

        return avg_losses

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, save_dir: str = 'checkpoints',
              save_every: int = 10, early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save model every N epochs
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary containing training history
        """
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            if self.tensorboard_logger:
                self.tensorboard_logger.set_epoch(epoch)

            # Training
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

            for batch in pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses)

                # Update progress bar
                pbar.set_postfix({
                    'gen_loss': f"{losses['total']:.4f}",
                    'disc_loss': f"{losses['disc_total']:.4f}"
                })

            # Average training losses
            avg_train_losses = dict()
            for key in epoch_losses[0].keys():
                avg_train_losses[key] = np.mean(
                    [losses[key] for losses in epoch_losses])

            # Validation
            val_losses = self.validate(val_loader, epoch)

            # Log learning rates and model parameters to TensorBoard
            if self.tensorboard_logger:
                # Log learning rates
                gen_lr = self.optimizer_generator.param_groups[0]['lr']
                disc_lr = self.optimizer_discriminator.param_groups[0]['lr']
                self.tensorboard_logger.log_learning_rates(
                    gen_lr, disc_lr, epoch)

                # Log model parameters (every 5 epochs to avoid too much data)
                if epoch % 5 == 0:
                    self.tensorboard_logger.log_model_parameters(
                        self.model, epoch)

            # Store losses
            self.train_losses.append(avg_train_losses)
            self.val_losses.append(val_losses)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(
                f"Train - Gen: {avg_train_losses['total']:.4f}, Disc: {avg_train_losses['disc_total']:.4f}")
            print(
                f"Val - Gen: {val_losses['total']:.4f}, Disc: {val_losses['disc_total']:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(
                    save_dir, f'nl2cm_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
                    'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

            # Early stopping
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                # Save best model
                best_path = os.path.join(save_dir, 'nl2cm_best.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
                    'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, best_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=get_device())
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_generator.load_state_dict(
            checkpoint['optimizer_generator_state_dict'])
        self.optimizer_discriminator.load_state_dict(
            checkpoint['optimizer_discriminator_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        return checkpoint['epoch']


    def close_tensorboard(self):
        """Close the TensorBoard logger."""
        if self.tensorboard_logger:
            self.tensorboard_logger.close()
    