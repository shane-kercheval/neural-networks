"""Architectures for various neural networks."""
from torchpy import net


class MLP(net.Sequential):
    """
    Simple Multi-Layer Perceptron with dynamic number of layers.

    Each hidden layer is initialied via he_init_scale and is followed by a BatchNorm layer and
    ReLU activation function. The output layer is initialized via glorot_init_scale and does not
    have an activation function.
    """

    def __init__(self, in_features: int, out_features: int, hidden_features: list[int]):
        """
        Initialize the MLP.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            hidden_features: List of hidden layer sizes.
        """
        layers = []
        prev_features = in_features
        for hidden_size in hidden_features:
            layers.append(net.Linear(
                in_features=prev_features,
                out_features=hidden_size,
                weight_init_scale=net.he_init_scale,
            ))
            layers.append(net.BatchNorm(hidden_size))
            layers.append(net.ReLU())
            prev_features = hidden_size
        layers.append(net.Linear(
            in_features=prev_features,
            out_features=out_features,
            weight_init_scale=net.glorot_init_scale,
        ))
        super().__init__(layers=layers)
