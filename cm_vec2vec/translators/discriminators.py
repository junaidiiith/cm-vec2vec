"""
Discriminator networks for adversarial training in CMVec2Vec
"""

import torch
import torch.nn as nn
from typing import Optional


class Discriminator(nn.Module):
    """
    Discriminator network for adversarial training.

    Distinguishes between real and fake (translated) embeddings to ensure
    translated embeddings match the target distribution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        depth: int = 3,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        activation: str = 'leaky_relu',
        weight_init: str = 'kaiming',
        output_activation: str = 'sigmoid'
    ):
        """
        Initialize the discriminator network.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            depth: Number of hidden layers
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            activation: Activation function ('leaky_relu', 'relu', 'silu')
            weight_init: Weight initialization method
            output_activation: Output activation ('sigmoid', 'linear')
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Choose activation function
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Choose output activation
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'linear':
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")

        # Build the network
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation)
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(self.output_activation)

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights(weight_init)

    def _initialize_weights(self, weight_init: str):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == 'kaiming':
                    nn.init.kaiming_normal_(
                        module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                elif weight_init == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                elif weight_init == 'orthogonal':
                    nn.init.orthogonal_(module.weight)
                else:
                    raise ValueError(
                        f"Unknown weight initialization: {weight_init}")
                module.bias.data.fill_(0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Discriminator scores of shape (batch_size, 1)
        """
        return self.network(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator that operates on different scales of the input.
    """

    def __init__(
        self,
        input_dim: int,
        scales: list = [1, 2, 4],
        **kwargs
    ):
        """
        Initialize multi-scale discriminator.

        Args:
            input_dim: Input embedding dimension
            scales: List of scale factors for downsampling
            **kwargs: Additional arguments for individual discriminators
        """
        super().__init__()

        self.scales = scales
        self.discriminators = nn.ModuleList()

        for scale in scales:
            if scale == 1:
                # No downsampling
                disc_input_dim = input_dim
            else:
                # Downsample by taking every scale-th element
                disc_input_dim = input_dim // scale

            self.discriminators.append(
                Discriminator(
                    input_dim=disc_input_dim,
                    **kwargs
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through all scale discriminators.

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            List of discriminator scores for each scale
        """
        outputs = []

        for i, scale in enumerate(self.scales):
            if scale == 1:
                # No downsampling
                disc_input = x
            else:
                # Downsample by taking every scale-th element
                disc_input = x[:, ::scale]

            outputs.append(self.discriminators[i](disc_input))

        return outputs


class ConditionalDiscriminator(Discriminator):
    """
    Discriminator that can be conditioned on additional information.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 0,
        **kwargs
    ):
        super().__init__(input_dim, **kwargs)
        self.cond_dim = cond_dim

        if cond_dim > 0:
            self.cond_projection = nn.Linear(cond_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional conditioning.

        Args:
            x: Input embeddings
            condition: Optional conditioning vector

        Returns:
            Discriminator scores
        """
        if condition is not None and self.cond_dim > 0:
            # Project condition to hidden dimension
            cond_proj = self.cond_projection(condition)

            # Apply condition to the first hidden layer
            x = self.network[0](x)  # First linear layer
            x = x + cond_proj  # Add condition
            x = self.network[1](x)  # Activation

            # Continue with rest of network
            for layer in self.network[2:]:
                x = layer(x)

            return x
        else:
            return self.network(x)
