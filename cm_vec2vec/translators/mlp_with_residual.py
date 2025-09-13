"""
MLP with residual connections for stable training
"""

import torch
import torch.nn as nn
from typing import Optional


def add_residual(input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Add residual connection between input and output.

    Args:
        input_x: Original input tensor
        x: Transformed tensor

    Returns:
        Sum of input and transformed tensors (with dimension matching)
    """
    if input_x.shape[1] < x.shape[1]:
        # Pad input to match output dimension
        padding = torch.zeros(
            x.shape[0], x.shape[1] - input_x.shape[1], device=x.device)
        input_x = torch.cat([input_x, padding], dim=1)
    elif input_x.shape[1] > x.shape[1]:
        # Truncate input to match output dimension
        input_x = input_x[:, :x.shape[1]]

    return x + input_x


class MLPWithResidual(nn.Module):
    """
    Multi-layer perceptron with residual connections for stable training.

    This is used as the backbone network in the shared latent space.
    """

    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        norm_style: str = 'layer',
        output_norm: bool = False,
        dropout: float = 0.1,
        activation: str = 'silu',
        weight_init: str = 'kaiming',
        use_residual: bool = True
    ):
        """
        Initialize the MLP with residual connections.

        Args:
            depth: Number of layers
            in_dim: Input dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension
            norm_style: Normalization style ('layer', 'batch', 'none')
            output_norm: Whether to apply normalization to output
            dropout: Dropout rate
            activation: Activation function ('silu', 'relu', 'gelu')
            weight_init: Weight initialization method
            use_residual: Whether to use residual connections
        """
        super().__init__()

        self.depth = depth
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_residual = use_residual

        # Choose activation function
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Choose normalization layer
        if norm_style == 'layer':
            norm_layer = nn.LayerNorm
        elif norm_style == 'batch':
            norm_layer = nn.BatchNorm1d
        elif norm_style == 'none':
            norm_layer = None
        else:
            raise ValueError(f"Unknown norm style: {norm_style}")

        # Build the network
        self.layers = nn.ModuleList()

        for layer_idx in range(self.depth):
            if layer_idx == 0:
                # Input layer
                current_hidden_dim = out_dim if self.depth == 1 else hidden_dim
                layer_components = [
                    nn.Linear(in_dim, current_hidden_dim),
                    self.activation
                ]
                if norm_layer is not None and self.depth > 1:
                    layer_components.append(norm_layer(current_hidden_dim))
                layer_components.append(nn.Dropout(dropout))

            elif layer_idx < self.depth - 1:
                # Hidden layers
                layer_components = [
                    nn.Linear(hidden_dim, hidden_dim),
                    self.activation
                ]
                if norm_layer is not None:
                    layer_components.append(norm_layer(hidden_dim))
                layer_components.append(nn.Dropout(dropout))

            else:
                # Output layer
                layer_components = [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                    self.activation,
                    nn.Linear(hidden_dim, out_dim)
                ]

            self.layers.append(nn.Sequential(*layer_components))

        # Output normalization
        if output_norm:
            self.output_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        else:
            self.output_norm = None

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
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if isinstance(module, nn.BatchNorm1d):
                    nn.init.normal_(module.weight, mean=1.0, std=0.02)
                    nn.init.normal_(module.bias, mean=0.0, std=0.02)
                else:  # LayerNorm
                    nn.init.constant_(module.bias, 0)
                    nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP with residual connections.

        Args:
            x: Input tensor of shape (batch_size, in_dim)

        Returns:
            Output tensor of shape (batch_size, out_dim)
        """
        for layer in self.layers:
            if self.use_residual and x.shape[1] == self.hidden_dim:
                # Apply residual connection
                input_x = x
                x = layer(x)
                x = add_residual(input_x, x)
            else:
                x = layer(x)

        # Apply output normalization if specified
        if self.output_norm is not None:
            x = self.output_norm(x)

        return x


class ConditionalMLPWithResidual(MLPWithResidual):
    """
    MLP with residual connections that supports conditioning.
    """

    def __init__(
        self,
        cond_dim: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cond_dim = cond_dim

        if cond_dim > 0:
            self.cond_projection = nn.Linear(cond_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional conditioning.

        Args:
            x: Input tensor
            condition: Optional conditioning vector

        Returns:
            Output tensor
        """
        if condition is not None and self.cond_dim > 0:
            # Project condition to hidden dimension
            cond_proj = self.cond_projection(condition)

            # Apply condition to the first layer
            x = self.layers[0][0](x)  # First linear layer
            x = x + cond_proj  # Add condition
            x = self.layers[0][1](x)  # Activation

            # Continue with rest of first layer
            for layer in self.layers[0][2:]:
                x = layer(x)

            # Continue with remaining layers
            for layer in self.layers[1:]:
                if self.use_residual and x.shape[1] == self.hidden_dim:
                    input_x = x
                    x = layer(x)
                    x = add_residual(input_x, x)
                else:
                    x = layer(x)
        else:
            # Standard forward pass
            for layer in self.layers:
                if self.use_residual and x.shape[1] == self.hidden_dim:
                    input_x = x
                    x = layer(x)
                    x = add_residual(input_x, x)
                else:
                    x = layer(x)

        # Apply output normalization if specified
        if self.output_norm is not None:
            x = self.output_norm(x)

        return x
