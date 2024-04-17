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


class CNN(net.Sequential):
    """
    Simple Convolutional Neural Network with dynamic number of layers.

    Each hidden layer is initialied via he_init_scale and is followed by a BatchNorm layer and
    ReLU activation function. The output layer is initialized via glorot_init_scale and does not
    have an activation function.
    """

    def __init__(self):
        """
        Initialize the CNN.

        Args:
            in_channels: Number of input channels.
            out_features: Number of output features.
            hidden_channels: List of hidden layer sizes.
        """
        layers = [
            net.Conv2D(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            net.ReLU(),
            net.MaxPool2D(kernel_size=2, stride=2),
            net.Conv2D(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            net.ReLU(),
            net.MaxPool2D(kernel_size=2, stride=2),
            net.Flatten(),
            net.Linear(
                in_features=32 * 7 * 7,
                out_features=128,
                weight_init_scale=net.he_init_scale,
            ),
            net.ReLU(),
            net.Linear(
                in_features=128,
                out_features=10,
                weight_init_scale=net.glorot_init_scale,
            ),
        ]
        super().__init__(layers=layers)
