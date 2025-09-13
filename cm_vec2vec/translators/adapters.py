"""
Adapter networks for translating between different embedding spaces
"""

import torch
import torch.nn as nn
from typing import Optional


class Adapter(nn.Module):
    """
    Adapter network that transforms embeddings between original space and shared latent space.

    This is the core component that maps embeddings from different domains (NL, CM, etc.)
    into a shared latent space where translation can occur.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = 'silu',
        weight_init: str = 'kaiming'
    ):
        """
        Initialize the adapter network.

        Args:
            in_dim: Input embedding dimension
            out_dim: Output embedding dimension (latent space dimension)
            hidden_dim: Hidden layer dimension
            depth: Number of hidden layers
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            activation: Activation function ('silu', 'relu', 'gelu')
            weight_init: Weight initialization method
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Choose activation function
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build the network
        layers = []

        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
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
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights(weight_init)

    def _initialize_weights(self, weight_init: str):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == 'kaiming':
                    nn.init.kaiming_normal_(
                        module.weight, a=0, mode='fan_in', nonlinearity='relu')
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
        Forward pass through the adapter.

        Args:
            x: Input embeddings of shape (batch_size, in_dim)

        Returns:
            Transformed embeddings of shape (batch_size, out_dim)
        """
        return self.network(x)


class ResidualAdapter(Adapter):
    """
    Adapter with residual connections for better gradient flow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        if self.use_residual and x.shape[1] == self.out_dim:
            return x + self.network(x)
        else:
            return self.network(x)


class ConditionalAdapter(Adapter):
    """
    Adapter that can be conditioned on additional information.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int = 0,
        **kwargs
    ):
        super().__init__(in_dim, out_dim, **kwargs)
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
            Transformed embeddings
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
