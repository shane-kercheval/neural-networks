
"""Includes building blocks for neural networks."""
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Callable, ClassVar
from contextlib import ContextDecorator
from numba import jit, prange
# from line_profiler import profile
import numpy as np
from numpy.random import default_rng


class training_mode(ContextDecorator):  # noqa: N801
    """A context manager to set the module in training mode temporarily."""\

    @classmethod
    def __enter__(cls: 'Module.training_mode'):
        Module.training = True
        return cls

    @classmethod
    def __exit__(cls: 'Module.training_mode', exc_type, exc_value, traceback):  # noqa: ANN001
        Module.training = False


class Module(ABC):
    """Base class for all neural network components/modules."""

    training: ClassVar[bool] = False

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
        Perform the backward pass of the module, calculating/returning the gradient with respect
        to input.

        Backpropagation is essentially an application of the chain rule, used to compute gradient
        of a loss function with respect to the weights and biases of a neural network. The
        gradient tells us how much (and in which direction) each parameter affects the loss
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

        ∂a/∂z is the gradient of the output with respect to the activation function, which is often
        straightforward to compute (e.g., for ReLU, sigmoid).

        ∂z/∂W is the gradient of the layer's output with respect to its weights, which typically
        involves the input to the layer, depending on whether it's fully connected, convolutional,
        etc.

        Args:
            grad_output: Gradient of the loss with respect to the output of this module.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this module.
        """

    def step(self, optimizer: Callable[[np.ndarray, np.ndarray], None]) -> None:
        """
        Update the parameters of the module using the gradient computed during
        backpropagation and the optimizer provided.

        Only applicable to modules that require gradient computation (TrainableParamsModule) but
        needs to be defined in the base class to avoid checking the type of module in the training
        loop.

        Args:
            optimizer (callable):
                The optimizer to use for updating the weights and biases. The optimizer is a
                function that takes the parameters (of the Module) as the first function parameter
                and the gradients as the second function parameter and updates the Module
                parameters.
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
    def _step(self, optimizer: Callable[[np.ndarray, np.ndarray], None]) -> None:
        """
        Update the weights and biases of the module using the gradient computed during
        backpropagation and the optimizer provided.

        Args:
            optimizer:
                The optimizer to use for updating the weights and biases. The optimizer takes
                the parameters as the first argument and the gradients as the second argument
                and updates the parameters in place.
        """

    def step(self, optimizer: Callable[[np.ndarray, np.ndarray], None]) -> None:
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
    in_features) for other activations. We use the ReLU variant here.

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

    def _step(self, optimizer: Callable[[np.ndarray, np.ndarray], None]) -> None:
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

    def __str__(self) -> str:
        """String representation of the Linear layer."""
        return f"Linear({self.weights.shape[0]}x{self.weights.shape[1]})"


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

    def __str__(self) -> str:
        """String representation of the ReLU layer."""
        return "ReLU()"


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

    def step(self, optimizer: Callable[[np.ndarray, np.ndarray], None]) -> None:
        """
        Update the parameters of the modules using the gradient computed during
        backpropagation.
        """
        for module in self.layers:
            module.step(optimizer)

    def __str__(self) -> str:
        """String representation of the Sequential container."""
        layers_str = "    " + ",\n    ".join(str(layer) for layer in self.layers)
        return dedent(f"""
        Sequential(
        {layers_str}
        )
        """).strip()


class BatchNorm(TrainableParamsModule):
    """
    Batch normalization layer.

    Normalizes the input to a layer for each mini-batch, which helps in accelerating the training
    process by reducing internal covariate shift.
    """

    def __init__(self, num_features: int, momentum: float = 0.9, seed: int = 42):
        """
        Initialize the batch normalization layer.

        Args:
            num_features:
                The number of features from the input tensors expected on the last axis.
            momentum:
                The momentum factor for the moving average of the mean and variance. Common values
                for momentum are between 0.9 and 0.99. These values determine how fast the running
                averages forget the oldest observed values. A higher momentum (closer to 1) means
                that older batches influence the running average more strongly and for longer,
                providing stability but making the running averages slower to react to changes in
                data distribution.
            seed:
                Random seed for reproducibility.
        """
        super().__init__()
        rng = default_rng(seed)
        self.eps = 1e-8  # added to the variance to avoid dividing by zero
        self.momentum = momentum
        # gamma and beta are learnable parameters of the layer
        # gamma is the scale parameter and allows the network to learn the optimal scale
        # for each feature
        # beta is the shift parameter and allows the network to learn the optimal mean for each
        # feature
        self.gamma = rng.normal(loc=1.0, scale=0.02, size=(1, num_features))
        self.beta = rng.normal(loc=0.0, scale=0.02, size=(1, num_features))
        # Moving statistics (not learnable parameters; only updated during training and used during
        # inference)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self.gamma_grad = None
        self.beta_grad = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for batch normalization.

        Args:
            x: Input data of shape (batch_size, num_features)

        Returns:
            np.ndarray: Normalized data.
        """
        # if training, calculate the batch mean and variance and update the running statistics
        # otherwise, normalize using the running statistics
        if Module.training:
            self.x = x
            # mean and variance for the batch
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)
            # Update the running mean and variance based on the momentum which is used to
            self.running_mean = self.momentum * self.running_mean \
                + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var \
                + (1 - self.momentum) * self.batch_var
            # Normalize the batch
            self.x_normalized = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            # normalize, scale, and then shift the data
            out = self.gamma * self.x_normalized + self.beta
        else:
            # Normalize using running statistics during inference
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for batch normalization.

        This method calculates the gradients of the batch normalization layer's output with respect
        to its inputs, as well as the gradients with respect to the learnable parameters gamma and
        beta

        Args:
            grad_output: Gradient of the loss with respect to the output of this layer (dL/dy).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer (dL/dx).
        """
        assert Module.training, "Backward pass should only be called during training."
        # Gradient with respect to parameters gamma (y) and beta (β)
        # dL/dy = o (dL/dy * x_norm)
        # Where x_norm = (x - batch_mean) / sqrt(batch_var + eps)
        self.gamma_grad = np.sum(grad_output * self.x_normalized, axis=0, keepdims=True)
        # dL/dβ = o (dL/dy)
        # Since β is added to the normalized input, its gradient is the sum of the incoming
        # gradients.
        self.beta_grad = np.sum(grad_output, axis=0, keepdims=True)
        # Gradient with respect to the input (dL/dx)
        num_samples = grad_output.shape[0]
        x_centered = self.x - self.batch_mean
        st_dev_inverse = 1 / np.sqrt(self.batch_var + self.eps)
        # Gradient with respect to the normalized input (x_norm)
        # dL/dx_norm = dL/dy * y
        dx_normalized = grad_output * self.gamma
        # Intermediate gradient for variance (o^2)
        # dL/do^2 = o (dL/dx_norm * (x - μ)) * -1/2 * o^-3
        dvar = np.sum(dx_normalized * x_centered, axis=0, keepdims=True) * -.5 * st_dev_inverse**3
        # Intermediate gradient for mean (μ)
        # dL/dμ = o (dL/dx_norm * -1/o) + dL/do^2 * o(-2 * (x - μ)/N)
        dmean = np.sum(dx_normalized * -st_dev_inverse, axis=0, keepdims=True) \
            + dvar * np.mean(-2. * x_centered, axis=0, keepdims=True)
        # Final gradient with respect to input x
        # dL/dx = dL/dx_norm / o + dL/do^2 * 2(x - μ)/N + dL/dμ/N
        return (dx_normalized * st_dev_inverse) \
            + (dvar * 2 * x_centered / num_samples) \
            + (dmean / num_samples)

    def _zero_grad(self) -> None:
        """Clear the gradient of the layer."""
        self.gamma_grad = None
        self.beta_grad = None

    def _step(self, optimizer: Callable[[np.ndarray, np.ndarray], None]) -> None:
        """
        Update the gamma and beta parameters using the provided optimizer.

        Args:
            optimizer (callable): The optimizer to use for updating the parameters.
        """
        assert Module.training
        assert self.gamma_grad is not None
        assert self.beta_grad is not None, "Gradient not calculated."
        optimizer(self.gamma, self.gamma_grad)
        optimizer(self.beta, self.beta_grad)
        self._zero_grad()

    def __str__(self) -> str:
        """String representation of the BatchNorm layer."""
        return f"BatchNorm({self.gamma.shape[1]}, momentum={self.momentum})"


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


@jit(nopython=True)
def _feature_map_dimensions(
        height: int,
        width: int,
        kernel_size: int,
        stride: int) -> tuple[int, int]:
    """
    Calculate the height and width of the feature map after applying the convolution operation.

    Args:
        height: Height of the input data (after padding if applied).
        width: Width of the input data (after padding if applied).
        kernel_size: Size of the kernel.
        stride: Stride of the convolution operation.
        padding: Padding applied to the input data.

    Returns:
        tuple[int, int]: Height and width of the feature map.
    """
    output_height = ((height - kernel_size) // stride) + 1
    output_width = ((width - kernel_size) // stride) + 1
    return output_height, output_width


@jit(nopython=True, parallel=True)
def convolve(
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        biases: np.ndarray,
        stride: int,
        kernel_size: int) -> None:
    """
    Perform the convolution operation on the input data using the weights and biases provided.
    Uses Numba to speed up the computation and parallelize the operation. The function updates the
    output (y) in place.
    """
    batch_size, _, height, width = x.shape
    out_channels = weights.shape[0]
    output_height, output_width = _feature_map_dimensions(height, width, kernel_size, stride)

    for sample_index in prange(batch_size):  # Parallel execution over batches
        for filter_index in prange(out_channels):  # Parallel execution over filters
            # apply the filter to the input data based on the stride and kernel size
            # this is what creates the feature maps (for each filter)
            for i in range(output_height):
                for j in range(output_width):
                    height_start = i * stride
                    height_end = height_start + kernel_size
                    width_start = j * stride
                    width_end = width_start + kernel_size
                    # this is basically what creates the figure on pg 483 of Hands on ML showing
                    # the receptive field of the filter
                    y[sample_index, filter_index, i, j] = np.sum(
                            x[sample_index, :, height_start:height_end, width_start:width_end] * \
                                weights[filter_index, :, :, :],
                        ) + biases[filter_index]


class Conv2D(TrainableParamsModule):
    """A 2D Convolutional Layer class that performs convolution operations on the input data."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            seed: int | None = None):
        """
        Initializes the Conv2D layer with given parameters and random weights.

        Args:
            in_channels:
                Number of channels in the input. This is the number of filters/feature maps in the
                previous layer.
            out_channels:
                Number of filters (producing different channels at the output). This is the number
                of filters in this layer. The number of filters is the same as the number of
                feature maps in the output. The act of applying a filter across the entire input
                data creates a feature map. In other words, each feature map is made up of the
                the same filter applied across the entire input data.
            kernel_size: Size of each filter (assumed square).
            stride:
                The horizontal and vertical step size between each application of the filter to
                the input data.
            padding:
                The number of zeros to add to the input data on each side of the height and width.
                For example a padding of 1 will add a row of zeros to the top, bottom, left, and
                right of the input data.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        rng = np.random.default_rng(seed)
        # self.weights stores the filters for the convolution operation
        # Each filter has shape (kernel_size, kernel_size)
        self.weights = rng.normal(
            loc=0,  # mean
            scale=0.1,  # standard deviation
            size=(out_channels, in_channels, kernel_size, kernel_size))  # Initialize weights
        self.biases = np.zeros(out_channels)
        self._zero_grad()

    def _zero_grad(self) -> None:
        """Resets gradients of weights and biases to zero."""
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the convolutional layer using the input data.

        Args:
            x (ndarray): Input data of shape (batch_size, in_channels, height, width).
        """
        # Pad the input
        # x is a 4D tensor with shape (batch_size, in_channels, height, width)
        x = np.pad(
            x,
            (
                (0, 0),  # no padding along the batch size dimension.
                (0, 0),  # no padding along the channels dimension.
                (self.padding, self.padding),  # padding before and after the height
                (self.padding, self.padding),  # padding before and after the width
            ),
            mode='constant',
        )
        batch_size, _, input_height, input_width = x.shape
        # the output is the result of applying the filters to the input data (the feature maps)
        output_height, output_width = _feature_map_dimensions(
            input_height, input_width, self.kernel_size, self.stride,
        )
        output = np.zeros((batch_size, self.out_channels, output_height, output_width))
        convolve(x, output, self.weights, self.biases, self.stride, self.kernel_size)
        if Module.training:
            self.x = x  # Store input for backpropagation
            self.output = output  # Store output for backpropagation
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the convolutional layer by computing the gradients with
        respect to the input, weights, and biases based on the output gradients provided.

        The convolution operation is a sliding window calculation over the input where each
        position of the window corresponds to a local region in the input that is multiplied by the
        filter weights and summed up to produce an output. During backpropagation,we need to
        compute:

        - `∂L/∂x`:
            The gradient of the loss with respect to the input (x), which is propagated back to
            previous layers.
        - `∂L/∂W`:
            The gradient of the loss with respect to the filter weights.
        - `∂L/∂b`:
            The gradient of the loss with respect to the biases.

        Args:
            grad_output:
                Gradient of the loss with respect to the output of this layer. This corresponds to
                `∂L/∂y` in calculus terms, where `y` is the output of the forward pass of this
                convolution layer.
        """
        _, _, output_height, output_width = grad_output.shape
        x_grad = np.zeros_like(self.x)
        self.weight_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.biases)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride  # Start index for the height dimension
                h_end = h_start + self.kernel_size  # End index for the height dimension
                w_start = j * self.stride  # Start index for the width dimension
                w_end = w_start + self.kernel_size  # End index for the width dimension
                for filter_index in range(self.out_channels):
                    # Extract the relevant receptive field from the input
                    x_slice = self.x[:, :, h_start:h_end, w_start:w_end]
                    # Update the gradient of the weights
                    self.weight_grad[filter_index] += np.sum(
                        x_slice * grad_output[:, [filter_index], i:i+1, j:j+1],
                        axis=0,
                    )
                    # Update the gradient of the biases
                    self.bias_grad[filter_index] += np.sum(grad_output[:, filter_index, i, j])
                    # Update the gradient of the input
                    x_grad[:, :, h_start:h_end, w_start:w_end] += self.weights[filter_index] * \
                        grad_output[:, [filter_index], i:i+1, j:j+1]

        if self.padding != 0:
            # Remove padding from gradient because it was added during forward pass
            x_grad = x_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return x_grad

    def _step(self, optimizer: callable) -> None:
        """
        Update the weights and biases using the computed gradients and the optimizer provided.

        Args:
            optimizer: The optimizer function to use for updating the parameters.
        """
        optimizer(self.weights, self.weight_grad)
        optimizer(self.biases, self.bias_grad)
        self._zero_grad()

    def __str__(self) -> str:
        """String representation of the Conv2D layer."""
        return f"Conv2D({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"  # noqa


@jit(nopython=True, parallel=True)
def maxpool2d_backward(
        grad_output: np.ndarray,
        grad_input: np.ndarray,
        input_data: np.ndarray,
        latest_output: np.ndarray,
        stride: int,
        kernel_size: int,
        out_height: int,
        out_width: int,
        ) -> None:
    """
    Perform the backward pass of the max pooling operation. The goal is to update `grad_input`
    which is the gradient of the loss with respect to the input data. grad_input is expected
    to be defined before calling this function and is modified in place.
    """
    for batch_index in prange(grad_output.shape[0]):  # Parallel execution over batches
        for channel_index in prange(grad_output.shape[1]):  # Parallel execution over channels
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * stride
                    h_end = h_start + kernel_size
                    w_start = j * stride
                    w_end = w_start + kernel_size
                    x_slice = input_data[batch_index, channel_index, h_start:h_end, w_start:w_end]
                    mask = (x_slice == latest_output[batch_index, channel_index, i, j])
                    grad_input[batch_index, channel_index, h_start:h_end, w_start:w_end] += \
                        mask * grad_output[batch_index, channel_index, i, j]


class MaxPool2D(Module):
    """
    Max pooling layer. Subsamples the input by taking the maximum value in each region defined by
    the kernel size and stride. This reduces the risk of overfitting and reduces the computational
    load by reducing the number of parameters, computations, and memory usage in the network.
    """

    def __init__(self, kernel_size: int, stride: int | None = None):
        """
        Initialize the MaxPool2D layer.

        Args:
            kernel_size:
                The size of the pooling region.
            stride:
                Stride of the pooling operation. If None, the stride is set to the kernel size,
                which is the default behavior of PyTorch.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the max pooling operation on the input data."""
        batch_size, channels, height, width = x.shape
        self.out_height = (height - self.kernel_size) // self.stride + 1
        self.out_width = (width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, channels, self.out_height, self.out_width))
        for i in range(self.out_height):
            for j in range(self.out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(x_slice, axis=(2, 3))
        if Module.training:
            self.x = x
            self.output = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Perform the backward pass of the max pooling layer."""
        assert Module.training
        grad_input = np.zeros_like(self.x)
        maxpool2d_backward(
            grad_output,
            grad_input,
            self.x,
            self.output,
            self.stride,
            self.kernel_size,
            self.out_height,
            self.out_width,
        )
        return grad_input

    def __str__(self):
        return f"MaxPool2D(kernel_size={self.kernel_size}, stride={self.stride})"


class Flatten(Module):
    """Flatten layer to reshape the input to a 1D array."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the Flatten layer."""
        # Save the input shape to reshape the gradient correctly in the backward pass
        self.input_shape = x.shape
        # Flatten the input except for the batch dimension
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of the Flatten layer."""
        # Reshape the gradient to the shape of the original input
        return grad_output.reshape(self.input_shape)

    def __str__(self):
        return "Flatten()"
