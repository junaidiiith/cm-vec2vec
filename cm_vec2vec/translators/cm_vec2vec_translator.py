"""
Main CMVec2Vec translator class for NL2CM translation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .adapters import Adapter, ConditionalAdapter
from .discriminators import Discriminator, ConditionalDiscriminator
from .mlp_with_residual import MLPWithResidual, ConditionalMLPWithResidual


class CMVec2VecTranslator(nn.Module):
    """
    Main translator class for converting between NL and CM embedding spaces.

    This class implements the vec2vec approach specifically for NL2CM translation,
    removing the multi-domain complexity and focusing on the core translation task.
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        adapter_depth: int = 3,
        backbone_depth: int = 4,
        dropout: float = 0.1,
        use_conditioning: bool = False,
        cond_dim: int = 0,
        normalize_embeddings: bool = True,
        weight_init: str = 'kaiming',
        activation: str = 'silu'
    ):
        """
        Initialize the CMVec2Vec translator.

        Args:
            embedding_dim: Dimension of NL and CM embeddings
            latent_dim: Dimension of the shared latent space
            hidden_dim: Hidden layer dimension for adapters and backbone
            adapter_depth: Depth of adapter networks
            backbone_depth: Depth of shared backbone network
            dropout: Dropout rate
            use_conditioning: Whether to use conditioning
            cond_dim: Dimension of conditioning vector
            normalize_embeddings: Whether to normalize output embeddings
            weight_init: Weight initialization method
            activation: Activation function
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_conditioning = use_conditioning
        self.cond_dim = cond_dim
        self.normalize_embeddings = normalize_embeddings

        # Create adapters for NL and CM
        # Input adapters: domain -> latent space
        if use_conditioning:
            self.nlt_adapter = ConditionalAdapter(
                in_dim=embedding_dim,
                out_dim=latent_dim,
                hidden_dim=hidden_dim,
                depth=adapter_depth,
                dropout=dropout,
                cond_dim=cond_dim,
                weight_init=weight_init,
                activation=activation
            )
            self.cmt_adapter = ConditionalAdapter(
                in_dim=embedding_dim,
                out_dim=latent_dim,
                hidden_dim=hidden_dim,
                depth=adapter_depth,
                dropout=dropout,
                cond_dim=cond_dim,
                weight_init=weight_init,
                activation=activation
            )
        else:
            self.nlt_adapter = Adapter(
                in_dim=embedding_dim,
                out_dim=latent_dim,
                hidden_dim=hidden_dim,
                depth=adapter_depth,
                dropout=dropout,
                weight_init=weight_init,
                activation=activation
            )
            self.cmt_adapter = Adapter(
                in_dim=embedding_dim,
                out_dim=latent_dim,
                hidden_dim=hidden_dim,
                depth=adapter_depth,
                dropout=dropout,
                weight_init=weight_init,
                activation=activation
            )

        # Output adapters: latent space -> domain
        if use_conditioning:
            self.nlt_output_adapter = ConditionalAdapter(
                in_dim=latent_dim,
                out_dim=embedding_dim,
                hidden_dim=hidden_dim,
                depth=adapter_depth,
                dropout=dropout,
                cond_dim=cond_dim,
                weight_init=weight_init,
                activation=activation
            )
            self.cmt_output_adapter = ConditionalAdapter(
                in_dim=latent_dim,
                out_dim=embedding_dim,
                hidden_dim=hidden_dim,
                depth=adapter_depth,
                dropout=dropout,
                cond_dim=cond_dim,
                weight_init=weight_init,
                activation=activation
            )
        else:
            self.nlt_output_adapter = Adapter(
                in_dim=latent_dim,
                out_dim=embedding_dim,
                hidden_dim=hidden_dim,
                depth=adapter_depth,
                dropout=dropout,
                weight_init=weight_init,
                activation=activation
            )
            self.cmt_output_adapter = Adapter(
                in_dim=latent_dim,
                out_dim=embedding_dim,
                hidden_dim=hidden_dim,
                depth=adapter_depth,
                dropout=dropout,
                weight_init=weight_init,
                activation=activation
            )

        # Shared backbone network
        if use_conditioning:
            self.backbone = ConditionalMLPWithResidual(
                depth=backbone_depth,
                in_dim=latent_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                dropout=dropout,
                cond_dim=cond_dim,
                weight_init=weight_init,
                activation=activation
            )
        else:
            self.backbone = MLPWithResidual(
                depth=backbone_depth,
                in_dim=latent_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                dropout=dropout,
                weight_init=weight_init,
                activation=activation
            )

        # Discriminators for adversarial training
        # Output space discriminators
        if use_conditioning:
            self.nlt_discriminator = ConditionalDiscriminator(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                weight_init=weight_init
            )
            self.cmt_discriminator = ConditionalDiscriminator(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                weight_init=weight_init
            )
        else:
            self.nlt_discriminator = Discriminator(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                weight_init=weight_init
            )
            self.cmt_discriminator = Discriminator(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                weight_init=weight_init
            )

        # Latent space discriminator
        if use_conditioning:
            self.latent_discriminator = ConditionalDiscriminator(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                weight_init=weight_init
            )
        else:
            self.latent_discriminator = Discriminator(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                weight_init=weight_init
            )

    def _get_latent_representation(
        self, embeddings: torch.Tensor, is_nlt: bool,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert embeddings to latent representation.

        Args:
            embeddings: Input embeddings
            is_nlt: True if NL embeddings, False if CM embeddings
            condition: Optional conditioning vector

        Returns:
            Latent representation
        """
        if self.use_conditioning and condition is not None:
            if is_nlt:
                return self.nlt_adapter(embeddings, condition)
            else:
                return self.cmt_adapter(embeddings, condition)
        else:
            if is_nlt:
                return self.nlt_adapter(embeddings)
            else:
                return self.cmt_adapter(embeddings)

    def _project_from_latent(
            self, latent_repr: torch.Tensor, is_nlt: bool,
            condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project latent representation to target domain.

        Args:
            latent_repr: Latent representation
            is_nlt: True if target is NL, False if target is CM
            condition: Optional conditioning vector

        Returns:
            Projected embeddings
        """
        if self.use_conditioning and condition is not None:
            if is_nlt:
                output = self.nlt_output_adapter(latent_repr, condition)
            else:
                output = self.cmt_output_adapter(latent_repr, condition)
        else:
            if is_nlt:
                output = self.nlt_output_adapter(latent_repr)
            else:
                output = self.cmt_output_adapter(latent_repr)

        if self.normalize_embeddings:
            output = output / output.norm(dim=1, keepdim=True)

        return output

    def translate_nlt_to_cmt(self, nlt_embeddings: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Translate NL embeddings to CM embeddings.

        Args:
            nlt_embeddings: NL embeddings
            condition: Optional conditioning vector

        Returns:
            Translated CM embeddings
        """
        # Convert to latent space
        latent_repr = self._get_latent_representation(
            nlt_embeddings, is_nlt=True, condition=condition)

        # Process through shared backbone
        if self.use_conditioning and condition is not None:
            latent_repr = self.backbone(latent_repr, condition)
        else:
            latent_repr = self.backbone(latent_repr)

        # Project to CM domain
        return self._project_from_latent(latent_repr, is_nlt=False, condition=condition)

    def translate_cmt_to_nlt(self, cmt_embeddings: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Translate CM embeddings to NL embeddings.

        Args:
            cmt_embeddings: CM embeddings
            condition: Optional conditioning vector

        Returns:
            Translated NL embeddings
        """
        # Convert to latent space
        latent_repr = self._get_latent_representation(
            cmt_embeddings, is_nlt=False, condition=condition)

        # Process through shared backbone
        if self.use_conditioning and condition is not None:
            latent_repr = self.backbone(latent_repr, condition)
        else:
            latent_repr = self.backbone(latent_repr)

        # Project to NL domain
        return self._project_from_latent(latent_repr, is_nlt=True, condition=condition)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        condition: Optional[torch.Tensor] = None,
        include_reps: bool = False,
        noise_level: float = 0.0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through the translator.

        Args:
            inputs: Dictionary with 'nlt' and 'cmt' keys containing embeddings
            condition: Optional conditioning vector
            include_reps: Whether to include latent representations in output
            noise_level: Level of noise to add during training

        Returns:
            Tuple of (reconstructions, translations, latent_representations)
        """
        reconstructions = {}
        translations = {}
        latent_reps = {} if include_reps else None

        # Process NL embeddings
        if 'nlt' in inputs:
            nlt_embeddings = inputs['nlt']

            # Add noise during training
            if self.training and noise_level > 0.0:
                nlt_embeddings = nlt_embeddings + \
                    torch.randn_like(nlt_embeddings) * noise_level
                if self.normalize_embeddings:
                    nlt_embeddings = nlt_embeddings / \
                        nlt_embeddings.norm(dim=1, keepdim=True)

            # Convert to latent space
            nlt_latent = self._get_latent_representation(
                nlt_embeddings, is_nlt=True, condition=condition)

            if include_reps:
                latent_reps['nlt'] = nlt_latent

            # Process through shared backbone
            if self.use_conditioning and condition is not None:
                processed_nlt_latent = self.backbone(nlt_latent, condition)
            else:
                processed_nlt_latent = self.backbone(nlt_latent)

            # Generate NL reconstruction
            reconstructions['nlt'] = self._project_from_latent(
                processed_nlt_latent, is_nlt=True, condition=condition)

        # Process CM embeddings
        if 'cmt' in inputs:
            cmt_embeddings = inputs['cmt']

            # Add noise during training
            if self.training and noise_level > 0.0:
                cmt_embeddings = cmt_embeddings + \
                    torch.randn_like(cmt_embeddings) * noise_level
                if self.normalize_embeddings:
                    cmt_embeddings = cmt_embeddings / \
                        cmt_embeddings.norm(dim=1, keepdim=True)

            # Convert to latent space
            cmt_latent = self._get_latent_representation(
                cmt_embeddings, is_nlt=False, condition=condition)

            if include_reps:
                latent_reps['cmt'] = cmt_latent

            # Process through shared backbone
            if self.use_conditioning and condition is not None:
                processed_cmt_latent = self.backbone(cmt_latent, condition)
            else:
                processed_cmt_latent = self.backbone(cmt_latent)

            # Generate CM reconstruction
            reconstructions['cmt'] = self._project_from_latent(
                processed_cmt_latent, is_nlt=False, condition=condition)

        # Generate cross-domain translations
        # NL -> CM translation
        if 'nlt' in inputs and 'nlt' in latent_reps:
            if self.use_conditioning and condition is not None:
                processed_nlt_latent = self.backbone(
                    latent_reps['nlt'], condition)
            else:
                processed_nlt_latent = self.backbone(latent_reps['nlt'])

            translations['cmt'] = self._project_from_latent(
                processed_nlt_latent, is_nlt=False, condition=condition)

        # CM -> NL translation
        if 'cmt' in inputs and 'cmt' in latent_reps:
            if self.use_conditioning and condition is not None:
                processed_cmt_latent = self.backbone(
                    latent_reps['cmt'], condition)
            else:
                processed_cmt_latent = self.backbone(latent_reps['cmt'])

            translations['nlt'] = self._project_from_latent(
                processed_cmt_latent, is_nlt=True, condition=condition)

        return reconstructions, translations, latent_reps

    def get_discriminator_scores(
        self,
        embeddings: Dict[str, torch.Tensor],
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get discriminator scores for embeddings.

        Args:
            embeddings: Dictionary with 'nlt' and 'cmt' keys containing embeddings
            condition: Optional conditioning vector

        Returns:
            Dictionary mapping discriminator names to scores
        """
        scores = {}

        if 'nlt' in embeddings:
            if self.use_conditioning and condition is not None:
                scores['nlt_output'] = self.nlt_discriminator(
                    embeddings['nlt'], condition)
            else:
                scores['nlt_output'] = self.nlt_discriminator(
                    embeddings['nlt'])

        if 'cmt' in embeddings:
            if self.use_conditioning and condition is not None:
                scores['cmt_output'] = self.cmt_discriminator(
                    embeddings['cmt'], condition)
            else:
                scores['cmt_output'] = self.cmt_discriminator(
                    embeddings['cmt'])

        return scores

    def get_discriminator_scores_for_translations(
        self,
        translations: Dict[str, torch.Tensor],
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get discriminator scores for translated embeddings.

        Args:
            translations: Dictionary with 'nlt' and 'cmt' keys containing translated embeddings
            condition: Optional conditioning vector

        Returns:
            Dictionary mapping discriminator names to fake scores
        """
        fake_scores = {}

        if 'nlt' in translations:
            if self.use_conditioning and condition is not None:
                fake_scores['nlt_output_fake'] = self.nlt_discriminator(
                    translations['nlt'], condition)
            else:
                fake_scores['nlt_output_fake'] = self.nlt_discriminator(translations['nlt'])

        if 'cmt' in translations:
            if self.use_conditioning and condition is not None:
                fake_scores['cmt_output_fake'] = self.cmt_discriminator(
                    translations['cmt'], condition)
            else:
                fake_scores['cmt_output_fake'] = self.cmt_discriminator(
                    translations['cmt'])

        return fake_scores

    def get_latent_discriminator_scores(
        self,
        latent_reps: Dict[str, torch.Tensor],
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get latent space discriminator scores.

        Args:
            latent_reps: Dictionary with 'nlt' and 'cmt' keys containing latent representations
            condition: Optional conditioning vector

        Returns:
            Latent discriminator scores
        """
        # Concatenate all latent representations
        all_latent = torch.cat(list(latent_reps.values()), dim=0)

        if self.use_conditioning and condition is not None:
            # Repeat condition for all latent representations
            batch_size = all_latent.shape[0]
            repeated_condition = condition.repeat(
                batch_size // condition.shape[0], 1)
            return self.latent_discriminator(all_latent, repeated_condition)
        else:
            return self.latent_discriminator(all_latent)
