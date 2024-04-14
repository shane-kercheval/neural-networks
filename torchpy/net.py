"""Includes building blocks for neural networks."""
from abc import ABC, abstractmethod
from typing import ClassVar
from contextlib import ContextDecorator
import numpy as np
from numpy.random import default_rng

class Module(ABC):
    """Base class for all neural network modules."""

    training: ClassVar[bool] = False

    class training_mode(ContextDecorator):  # noqa: N801
        """A context manager to set the module in training mode temporarily."""\

        @classmethod
        def __enter__(cls: 'Module.training_mode'):
            cls.previous_mode = Module.training
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
        Backpropagation is essentially an application of the chain rule, used to compute gradient
        of a loss function with respect to the weights and biases of a neural network. The
        gradient tell us how much (and in which direction) each parameter affects the loss
        function (which we hope to minimize). Each layer's backward pass computes the gradients
        (i.e. the partial derivatives) of the loss function relative to its inputs and parameters,
        using the gradient that flows back from the subsequent layers.

        ∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W

        ∂L/∂W is the gradient (partial derivative) of the loss function (L) with respect to the
        weights (W) in this layer. This is what tells us how to adjust the weights of this layer
        to minimize the loss function.

        ∂L/∂a (i.e. grad_output) is the gradient of the loss function (L) with respect to the
        output of the layer. This is calculated in the subsequent layer (or directly from the loss
        function if it's the final layer) and passed back to this layer.

        ∂a/∂z is the derivative of the activation function, which is often straightforward to
        compute (e.g., for ReLU, sigmoid).

        ∂z/∂W is the derivative of the layer's output with respect to its weights, which typically
        involves the input to the layer, depending on whether it's fully connected, convolutional,
        etc.

        Perform the backward pass of the module, calculating gradient with respect to input.

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
        rng = default_rng()
        self.weights = rng.standard_normal((in_features, out_features)) * np.sqrt(2. / in_features)
        self.biases = np.zeros((1, out_features))
        self._zero_grad()

    def _zero_grad(self) -> None:
        """Clear the gradient of the layer."""
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the Linear layer.

        Args:
            x: Input data of shape (batch_size, in_features).

        Returns:
            np.ndarray: Output of the layer of shape (batch_size, out_features).
        """
        if Module.training:
            self.x = x  # Cache the input for backward pass
        return x @ self.weights + self.biases

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Returns the gradient with respect to the input.

        The Linear layer is defined by: output=input*weights + biases i.e. y=xW+b

        Where,
        - `x` is the input matrix (passed to forward) with shape (batch size, in_features)
        - `W` is the weight matrix with shape (in_features, out_features)
        - `b` is the bias vector with shape (1, out_features)
        - `y` is the output matrix with shape (batch size, out_features)

        Our goal is to compute the gradient of the loss function with respect to all of the
        parameters of this layer, and to propagate the gradient of the loss function backward
        to previous layers in the network:

        - `∂L/∂W`: The gradient of the loss function with respect to the weights
        - `∂L/∂b`: gradient of the loss function with respect to the biases
        - `∂L/∂x`: The gradient of the loss function with respect to the input, which is necessary
            to propagate the back to previous layers.
        - `∂L/∂y`: grad_output: the gradient of the loss function with respect to the output y

        Gradient of L with respect to the weights is calculated as:

            - `∂L/∂W = ∂L/∂y * ∂y/∂W`, where ∂L/∂y is grad_output
            - `∂y/∂W`: (the partial derivative of y with respect to W) means we treat all other
                variables (the bias and inputs) as constants) is `x`, because `y = xW + b` and the
                derivative b is a constant (which is 0) and the derivative of W with respect to W
                is 1, so the derivative of y with respect to W is x.
            - so `∂L/∂W = ∂L/∂y * ∂y/∂W` = grad_output * x, but we need to align the dimensions
                correctly for matrix multiplication, so we transpose x to get the correct shape.
                The dimension of self.x.T is (in_features, batch_size), and the dimension of
                grad_output is (batch_size, out_features) so the matrix multiplication is
                (in_features, batch_size) @ (batch_size, out_features).

        Gradient of L with respect to the biases is calculated as:

            - `∂L/∂b = ∂L/∂y * ∂y/∂b`, where ∂L/∂y is grad_output
            - `∂y/∂b  is 1, because `y = xW + b` and W and x are treated as constants, so the\
                derivative of y with respect to b is simply 1.
            - So ∂L/∂b = ∂L/∂y * ∂y/∂b = grad_output * 1 = grad_output

        Gradient of L with respect to the input (x) is calculated as:

                - `∂L/∂x = ∂L/∂y * ∂y/∂x`, where ∂L/∂y is grad_output
                - `∂y/∂x` is W, because `y = xW + b` where b and W are treated as a constants
                - so `∂L/∂x = ∂L/∂y * ∂y/∂x = grad_output * W`

        Args:
            grad_output:
                (∂L/∂y) Gradient of the loss with respect to the output of this layer. Calculated
                in the next layer and passed to this layer during backpropagation.
        """
        assert Module.training
        assert self.x.shape[0] == grad_output.shape[0], "Mismatch in batch size"
        assert self.x.shape[1] == self.weights.shape[0], "Input features do not match weights configuration"  # noqa
        self.weight_grad = self.x.T @ grad_output
        self.bias_grad = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.weights.T  # gradient with respect to input (x)

    def step(self, optimizer: callable) -> None:
        """
        Update the weights and biases of the layer using the gradient computed during
        backpropagation and the optimizer provided.
        """
        assert self.weight_grad is not None
        assert self.bias_grad is not None
        optimizer.update(self.weights, self.weight_grad)
        optimizer.update(self.biases, self.bias_grad)
        self._zero_grad()
