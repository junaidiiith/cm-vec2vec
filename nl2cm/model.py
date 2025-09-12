"""
NL2CM Model Architecture

This module implements the NL2CM translation model based on the vec2vec approach.
The model uses adapters and a shared backbone to translate between NL and CM embedding spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class Adapter(nn.Module):
    """
    Adapter module that maps embeddings to/from the shared latent space.

    This is a simple MLP with residual connections, LayerNorm, and SiLU activation.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512,
                 depth: int = 3, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Build the network
        layers = []
        current_dim = input_dim

        for i in range(depth):
            next_dim = hidden_dim if i < depth - 1 else output_dim
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim) if i < depth - 1 else nn.Identity(),
                nn.SiLU() if i < depth - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < depth - 1 else nn.Identity()
            ])
            current_dim = next_dim

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapter."""
        return self.network(x)


class SharedBackbone(nn.Module):
    """
    Shared backbone that processes embeddings in the latent space.

    This module refines the latent representation and can be conditioned
    on the target modeling language.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 512,
                 depth: int = 4, dropout: float = 0.1,
                 use_conditioning: bool = False, cond_dim: int = 0):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_conditioning = use_conditioning
        self.cond_dim = cond_dim

        # Input dimension includes conditioning if used
        input_dim = latent_dim + cond_dim if use_conditioning else latent_dim

        # Build the network with residual connections
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the shared backbone.

        Args:
            x: Input tensor of shape (batch_size, latent_dim)
            condition: Optional condition tensor of shape (batch_size, cond_dim)

        Returns:
            Output tensor of shape (batch_size, latent_dim)
        """
        if self.use_conditioning and condition is not None:
            x = torch.cat([x, condition], dim=-1)

        # Apply layers with residual connections
        for layer in self.layers:
            residual = x if x.shape[-1] == self.hidden_dim else None
            x = layer(x)
            if residual is not None:
                x = x + residual

        # Output projection
        return self.output_proj(x)


class Discriminator(nn.Module):
    """
    Discriminator for adversarial training.

    This discriminator distinguishes between real and fake embeddings
    in both the output space and latent space.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 depth: int = 3, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Build the network
        layers = []
        current_dim = input_dim

        for i in range(depth):
            next_dim = hidden_dim if i < depth - 1 else 1
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim) if i < depth - 1 else nn.Identity(),
                nn.LeakyReLU(0.2) if i < depth - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < depth - 1 else nn.Identity()
            ])
            current_dim = next_dim

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator."""
        return self.network(x)


class NL2CMTranslator(nn.Module):
    """
    Main NL2CM translation model.

    This model implements the vec2vec approach for translating between
    Natural Language and Conceptual Model embedding spaces.
    """

    def __init__(self, embedding_dim: int, latent_dim: int = 256,
                 hidden_dim: int = 512, adapter_depth: int = 3,
                 backbone_depth: int = 4, dropout: float = 0.1,
                 use_conditioning: bool = False, cond_dim: int = 0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_conditioning = use_conditioning
        self.cond_dim = cond_dim

        # Input adapters: map embeddings to latent space
        self.nlt_adapter = Adapter(
            embedding_dim, latent_dim, hidden_dim, adapter_depth, dropout)
        self.cmt_adapter = Adapter(
            embedding_dim, latent_dim, hidden_dim, adapter_depth, dropout)

        # Shared backbone: process embeddings in latent space
        self.backbone = SharedBackbone(latent_dim, hidden_dim, backbone_depth,
                                       dropout, use_conditioning, cond_dim)

        # Output adapters: map from latent space back to embedding space
        self.nlt_output_adapter = Adapter(
            latent_dim, embedding_dim, hidden_dim, adapter_depth, dropout)
        self.cmt_output_adapter = Adapter(
            latent_dim, embedding_dim, hidden_dim, adapter_depth, dropout)

        # Discriminators for adversarial training
        self.nlt_discriminator = Discriminator(
            embedding_dim, hidden_dim, 3, dropout)
        self.cmt_discriminator = Discriminator(
            embedding_dim, hidden_dim, 3, dropout)
        self.latent_discriminator = Discriminator(
            latent_dim, hidden_dim, 3, dropout)

    def forward(self, batch: Dict[str, torch.Tensor],
                condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the translator.

        Args:
            batch: Dictionary containing 'nlt' and 'cmt' embeddings
            condition: Optional condition tensor for conditioning

        Returns:
            Dictionary containing translations, reconstructions, and latents
        """
        nlt_emb = batch['nlt']
        cmt_emb = batch['cmt']

        # Get latents
        nlt_latent = self.nlt_adapter(nlt_emb)
        cmt_latent = self.cmt_adapter(cmt_emb)

        # Process through shared backbone
        nlt_processed = self.backbone(nlt_latent, condition)
        cmt_processed = self.backbone(cmt_latent, condition)

        # Generate outputs
        nlt_to_cmt = self.cmt_output_adapter(nlt_processed)
        cmt_to_nlt = self.nlt_output_adapter(cmt_processed)

        # Reconstructions
        nlt_recon = self.nlt_output_adapter(nlt_processed)
        cmt_recon = self.cmt_output_adapter(cmt_processed)

        # Normalize outputs
        nlt_to_cmt = F.normalize(nlt_to_cmt, p=2, dim=1)
        cmt_to_nlt = F.normalize(cmt_to_nlt, p=2, dim=1)
        nlt_recon = F.normalize(nlt_recon, p=2, dim=1)
        cmt_recon = F.normalize(cmt_recon, p=2, dim=1)

        return {
            'nlt_to_cmt': nlt_to_cmt,
            'cmt_to_nlt': cmt_to_nlt,
            'nlt_recon': nlt_recon,
            'cmt_recon': cmt_recon,
            'nlt_latent': nlt_latent,
            'cmt_latent': cmt_latent,
            'nlt_processed': nlt_processed,
            'cmt_processed': cmt_processed
        }

    def translate_nlt_to_cmt(self, nlt_emb: torch.Tensor,
                             condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Translate NL embeddings to CM embeddings."""
        with torch.no_grad():
            latent = self.nlt_adapter(nlt_emb)
            processed = self.backbone(latent, condition)
            output = self.cmt_output_adapter(processed)
            return F.normalize(output, p=2, dim=1)

    def translate_cmt_to_nlt(self, cmt_emb: torch.Tensor,
                             condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Translate CM embeddings to NL embeddings."""
        with torch.no_grad():
            latent = self.cmt_adapter(cmt_emb)
            processed = self.backbone(latent, condition)
            output = self.nlt_output_adapter(processed)
            return F.normalize(output, p=2, dim=1)

    def get_discriminator_outputs(self, batch: Dict[str, torch.Tensor],
                                  outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get discriminator outputs for adversarial training."""
        nlt_emb = batch['nlt']
        cmt_emb = batch['cmt']

        # Output space discriminators
        nlt_real = self.nlt_discriminator(nlt_emb)
        nlt_fake = self.nlt_discriminator(outputs['cmt_to_nlt'])
        cmt_real = self.cmt_discriminator(cmt_emb)
        cmt_fake = self.cmt_discriminator(outputs['nlt_to_cmt'])

        # Latent space discriminator
        latent_real = self.latent_discriminator(outputs['cmt_processed'])
        latent_fake = self.latent_discriminator(outputs['nlt_processed'])

        return {
            'nlt_real': nlt_real,
            'nlt_fake': nlt_fake,
            'cmt_real': cmt_real,
            'cmt_fake': cmt_fake,
            'latent_real': latent_real,
            'latent_fake': latent_fake
        }
