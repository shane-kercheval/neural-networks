
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
            Module.training = True
            return cls

        @classmethod
        def __exit__(cls: 'Module.training_mode', exc_type, exc_value, traceback):  # noqa: ANN001
            Module.training = False


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
            grad_output: Gradient of the loss with respect to the output of this module.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this module.
        """
        raise NotImplementedError

    def step(self, optimizer: callable) -> None:
        """
        Update the parameters of the module using the gradient computed during
        backpropagation and the optimizer provided.

        Only applicable to modules that require gradient computation (GradientModule) but needs to
        be defined in the base class to avoid checking the type of module in the training loop.

        Args:
            optimizer (callable): The optimizer to use for updating the weights and biases.
        """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Allows the module to be called like a function and directly use the forward pass.

        Args:
            x: Input data.

        Returns:
            np.ndarray: Output of the module.
        """
        return self.forward(x)


class TrainableParamsModule(Module):
    """
    Base class for modules that update their parameters (e.g. weights/biases) via
    backpropegation.
    """

    @abstractmethod
    def _zero_grad(self) -> None:
        """Clear the gradient of the module."""

    @abstractmethod
    def _step(self, optimizer: callable) -> None:
        """
        Update the weights and biases of the module using the gradient computed during
        backpropagation and the optimizer provided.

        Args:
            optimizer (callable): The optimizer to use for updating the weights and biases.
        """

    def step(self, optimizer: callable) -> None:
        """
        Update the weights and biases of the module using the gradient computed during
        backpropagation and the optimizer provided.

        Args:
            optimizer (callable): The optimizer to use for updating the weights and biases.
        """
        assert Module.training
        self._step(optimizer)
        self._zero_grad()


def he_init_scale(in_features: int, _: int | None = None) -> float:
    """
    He initialization is a way to initialize the weights of a neural network in a way that
    prevents the signal from vanishing or exploding as it passes through the network.

    The factor is calculated as sqrt(2. / in_features) for ReLU activations, and sqrt(1. /
    in_features) for other activations.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Returns:
        float: The scaling factor for the weights.
    """
    return np.sqrt(2. / in_features)


def glorot_init_scale(in_features: int, out_features: int) -> float:
    """
    Glorot initialization (also known as Xavier initialization) is a way to initialize the weights
    of a neural network in a way that prevents the signal from vanishing or exploding as it passes
    through the network.

    The factor is calculated as sqrt(6. / (in_features + out_features)).

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Returns:
        float: The scaling factor for the weights.
    """
    return np.sqrt(6. / (in_features + out_features))


class Linear(TrainableParamsModule):
    """Linear (fully connected) layer."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            weight_init_scale: callable = he_init_scale,
            seed: int = 42):
        """
        Initialize a Linear (fully connected) layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            weight_init_scale: Function to calculate the scaling factor for the weights.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        # TODO initialize the weights using He initialization (np.sqrt(2. / in_features)) is good
        # practice for layers followed by ReLU activations, as it helps in maintaining a balance
        # in the variance of activations across layers. If you plan to use other types of
        # activations, consider adjusting the initialization accordingly.
        rng = default_rng(seed)
        init_scale = weight_init_scale(in_features, out_features)
        self.weights = rng.normal(loc=0, scale=init_scale, size=(in_features, out_features))
        self.biases = np.zeros((1, out_features))
        self._zero_grad()

    def _zero_grad(self) -> None:
        """Clear the gradient of the layer."""
        self.weight_grad = None
        self.bias_grad = None

    def _step(self, optimizer: callable) -> None:
        """
        Update the weights and biases of the module using the gradient computed during
        backpropagation and the optimizer provided.

        Args:
            optimizer (callable): The optimizer to use for updating the weights and biases.
        """
        assert self.weight_grad is not None
        assert self.bias_grad is not None
        optimizer(self.weights, self.weight_grad)
        optimizer(self.biases, self.bias_grad)
        self._zero_grad()

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
        assert self.x.shape[1] == self.weights.shape[0], \
            "Input features do not match weights configuration"
        self.weight_grad = self.x.T @ grad_output
        self.bias_grad = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.weights.T  # gradient with respect to input (x)


class ReLU(Module):
    """ReLU activation function."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of ReLU."""
        output = np.maximum(0, x)
        if Module.training:
            self.output = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of ReLU.

        The ReLU function is defined as:

            `f(x) = max(0, x)`

        The derivative of ReLU:

            `f'(x) = 1 if x > 0 else 0`

        Therefore:
            `∂L/∂x = ∂L/∂f * ∂f/∂x`, where
                - `∂f/∂x` = f'(x)
                - `∂L/∂f` is the gradient of the loss function with respect to the output of this
                    layer, which is passed as grad_output.

        Args:
            grad_output:
                (∂L/∂y) Gradient of the loss with respect to the output of this layer. Calculated
                in the next layer and passed to this layer during backpropagation.
        """
        assert Module.training
        assert self.output is not None, "Forward pass not called before backward pass"
        return grad_output * (self.output > 0)  # Element-wise multiplication


class CrossEntropyLoss(Module):
    """
    Combines Softmax and Cross-Entropy loss into one single class for stability and
    efficiency.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logits = None
        self.targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Computes the cross-entropy loss from logits and targets directly.

        Args:
            logits: Logits array (before softmax).
            targets: Array of target class indices.

        Returns:
            float: Computed cross-entropy loss.
        """
        if Module.training:
            self.logits = logits
            self.targets = targets

        # Compute the softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        clipped_probs = np.clip(probabilities[np.arange(len(targets)), targets], 1e-15, 1.0)
        log_probs = -np.log(clipped_probs)
        return np.mean(log_probs)

    def backward(self) -> np.ndarray:
        """Computes and returns the gradient of the loss with respect to the logits."""
        assert Module.training
        assert self.logits is not None
        assert self.targets is not None, "Forward pass not called before backward."

        # Calculate the softmax probabilities (again)
        exp_logits = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        softmax_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Initialize gradient of logits
        grad_logits = softmax_probs.copy()
        # Subtract 1 from the probabilities of the correct classes
        # This step directly corresponds to the derivative of the loss function
        # with respect to each class probability:
        # ∂L/∂p_k = p_k - y_k, where y_k is 1 for the correct class and 0 otherwise.
        # Since softmax_probs contains p_k for all k, and we need to subtract 1 for the correct
        # class; this operation modifies the gradient correctly only for the indices of the correct
        # classes.
        grad_logits[np.arange(len(self.targets)), self.targets] -= 1

        # Average the gradients over the batch
        # The division by the number of examples handles the mean operation in the loss
        # calculation, as the loss is the average over all examples in the batch.
        grad_logits /= len(self.targets)

        # Avoid reusing the gradients in the next iteration
        self.logits = None
        self.targets = None
        return grad_logits

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Forward pass for computing the cross-entropy loss.

        Args:
            logits:
                Logits of the model, which are the raw, unnormalized predictions from the model
                (before softmax is applied).
            targets:
                Array of integers where each element is a class index which is the ground truth
                label for the corresponding input.
        """
        return self.forward(logits, targets)


class Softmax(Module):
    """Softmax activation function."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of Softmax.

        Uses `x - max(x)` in exponentiation which is a common enhancement to improve numerical
        stability during computation. If the elements of x are very large or very small,
        exponentiating them can lead to numerical issues. Exponentiating large numbers can lead to
        extremely large outputs, which can cause computational issues due to overflow (numbers too
        large to represent in a floating-point format). Exponentiating very negative numbers can
        result in values that are very close to zero, leading to underflow (numbers too small to
        represent accurately in a floating-point format).
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = exp_x / exp_x.sum(axis=1, keepdims=True)
        if Module.training:
            self.output = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of Softmax.

        The Softmax function for a vector z is defined as (the subtraction of max(x) in the
        exponentiation in the forward pass does not affect the derivative):

            `S(z)_i = exp(z_i) / sum(exp(z_j))`

        The derivative of Softmax is:

            `∂S_i/∂z_j = S_i * (δ_ij - S_j)`, where `δ_ij` is the Kronecker delta,
                which is 1 if i = j and 0 otherwise.

        For a given ∂L/∂S, where L is the loss function:

            ∂L/∂z_j = sum(∂L/∂S_i * ∂S_i/∂z_j)
            = sum(∂L/∂S_i * S_i * (δ_ij - S_j))
            = S_j * ( ∂L/∂S_j - sum(S_i * ∂L/∂S_i) )

            where ∂L/∂S_j is the gradient of the loss with respect to the output i.e. grad_output.

        Args:
            grad_output:
                (∂L/∂y) Gradient of the loss with respect to the output of this layer. Calculated
                in the next layer and passed to this layer during backpropagation.
        """
        assert Module.training
        assert self.output is not None, "Forward pass not called before backward pass"
        s = self.output
        return s * (grad_output - np.sum(grad_output * s, axis=1, keepdims=True))


class Sequential(Module):
    """Sequential container to stack Neural Network modules."""

    def __init__(self, layers: list[Module]):
        super().__init__()
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the Sequential container."""
        for module in self.layers:
            x = module(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of the Sequential container."""
        for module in reversed(self.layers):
            grad_output = module.backward(grad_output)
        return grad_output

    def step(self, optimizer: callable) -> None:
        """
        Update the parameters of the modules using the gradient computed during
        backpropagation.
        """
        for module in self.layers:
            module.step(optimizer)


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate: float):
        """
        Args:
            learning_rate: The learning rate (multiplier) to use for the optimizer.
        """
        self.learning_rate = learning_rate

    def __call__(self, parameters: np.ndarray, grads: np.ndarray) -> None:
        """Update the parameters using the gradients and learning rate."""
        parameters -= self.learning_rate * grads
