"""Neural network definitions used for prediction."""

from typing import List

from torch import nn


class BoundedMultiLayerPerceptron(nn.Module):
    """Small MLP with sigmoid scaling to bound output in [0, 1]."""

    def __init__(
        self,
        input_dimension: int,
        hidden_layer_sizes: List[int],
        dropout_rate: float = 0.10,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous_size = input_dimension
        for hidden_size in hidden_layer_sizes:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.network_body = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):  # type: ignore[override]
        return self.sigmoid(self.network_body(inputs))

    @property
    def body(self) -> nn.Sequential:
        """Backward-compatible accessor for older checkpoints."""
        return self.network_body
