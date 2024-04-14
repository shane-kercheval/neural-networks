"""TODO."""
from abc import ABC, abstractmethod
from typing import List, Tuple, ClassVar
from contextlib import ContextDecorator
import numpy as np

class Module(ABC):
    """Base class for all neural network modules."""

    training: ClassVar[bool] = False

    class training_mode(ContextDecorator):  # noqa: N801
        """A context manager to set the module in training mode temporarily."""\

        @classmethod
        def __enter__(cls: 'Module.training_mode'):
            Module.training = True
            return cls

        @classmethod
        def __exit__(cls: 'Module.training_mode', exc_type, exc_value, traceback):  # noqa: ANN001
            Module.training = cls.previous_mode


    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the module.

        Args:
            x: Input data.
        """

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backpropagation is essentially an application of the chain rule, used to compute gradients
        of a loss function with respect to the weights and biases of a neural network. The
        gradients tell us how much (and in which direction) each parameter affects the loss
        function. Each layer's backward pass computes the gradients (partial derivatives) of the
        loss function relative to its inputs and parameters, using the gradients that flow back
        from the subsequent layers.

        ∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W

        ∂L/∂W is the gradient (partial derivative) of the loss function (L) with respect to the
        weights (W) in this layer. This is what we want to compute.

        ∂L/∂a (i.e. grad_output) is the gradient of the loss function (L) with respect to the
        output of the layer. This is provided by the subsequent layer (or directly from the loss
        function if it's the final layer).

        ∂a/∂z is the derivative of the activation function, which is often straightforward to
        compute (e.g., for ReLU, sigmoid).

        ∂z/∂W is the derivative of the layer's output with respect to its weights, which typically
        involves the input to the layer, depending on whether it's fully
        connected, convolutional, etc.

        Perform the backward pass of the module, calculating gradients with respect to input.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output of this module.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this module.
        """
        raise NotImplementedError


    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Allows the module to be called like a function and directly use the forward pass.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the module.
        """
        return self.forward(x)


class Linear(Module):
    """Linear (fully connected) layer."""

    def __init__(self, in_features: int, out_features: int):
        """
        Initialize a Linear (fully connected) layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super().__init__()
        # TODO initialize the weights using He initialization (np.sqrt(2. / in_features)) is good
        # practice for layers followed by ReLU activations, as it helps in maintaining a balance
        # in the variance of activations across layers. If you plan to use other types of
        # activations, consider adjusting the initialization accordingly.
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.biases = np.zeros((1, out_features))
        self.params = [self.weights, self.biases]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the Linear layer.

        Args:
            x (np.ndarray): Input data of shape (batch_size, in_features).

        Returns:
            np.ndarray: Output of the layer of shape (batch_size, out_features).
        """
        self.x = x  # Cache the input for backward pass
        return x @ self.weights + self.biases

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of the Linear layer. Returns the gradient with respect to the input.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output of this layer.
        """
        assert Module.training
        assert self.x.shape[0] == grad_output.shape[0], "Mismatch in batch size"
        assert self.x.shape[1] == self.weights.shape[0], "Input features do not match weights configuration"
        # The goal of the backward pass is to compute the gradient of the loss with respect to
        # the input (dx), weights (dW), and biases (db).
        # The gradients of the loss with respect to the weights and biases are computed using the
        # dL_dW = ∂L/∂W = ∂L/∂z * ∂z/∂W
        # where z is the output of this layer
        # and ∂L/∂z is the gradient of the loss with respect to the output of the layer, which is
        # passed as an argument to this method (grad_output). In other words it's the gradient
        # flowing into this layer that was computed during the backpropagation of the previous
        # compution (subsequent layer since we are flowing backward).
        # ∂z/∂W is the gradient of the output of this layer with respect to the weights
        # Again, z is the equation for this layer (i.e. the forward pass) which is z = x*W + b
        # and so ∂z/∂W = x because we treat x and b as constants when computing the derivative
        # and the derivative of W with respect to W is 1 and the derivative of b (which is a
        # constant) is 0 so ∂z/∂W = x*1 + 0 = x.
        # ∂L/∂z = grad_output
        # ∂z/∂W = x
        # ∂L/∂W = grad_output * x
        self.weight_grads = self.x.T @ grad_output
        self.bias_grads = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.weights.T  # gradient with respect to input (x)

    def step(self, optimizer: callable) -> None:
        """
        Update the weights and biases of the layer using the gradients computed during
        backpropagation and the optimizer provided.
        """
        assert self.weight_grads is not None
        assert self.bias_grads is not None
        optimizer.update(self.weights, self.weight_grad)
        optimizer.update(self.biases, self.bias_grad)
        # Clear gradients after updating
        self.weight_grad = None
        self.bias_grad = None
