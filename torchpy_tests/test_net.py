"""Tests the neural network modules in torchpy.net."""
import numpy as np
import pytest
from torchpy import net


@pytest.fixture()
def linear_small() -> net.Linear:
    """Set up a Linear layer with 4 input features and 3 output features."""
    input_features = 4
    output_features = 3
    seed = 42
    return net.Linear(input_features, output_features, seed=seed)


@pytest.fixture()
def sequential_small() -> net.Sequential:
    """Set up a Sequential model with a Linear layer and ReLU activation."""
    input_features = 4
    output_features = 3
    return net.Sequential([
        net.Linear(input_features, output_features, seed=42),
        net.ReLU(),
    ])

def test_linear_forward(linear_small):  # noqa
    rng = np.random.default_rng()
    x = rng.standard_normal((2, 4))  # batch_size=2, input_features=4
    output = linear_small(x)
    assert output.shape == (2, 3)

def test_linear_backward(linear_small):  # noqa
    linear = linear_small
    rng = np.random.default_rng()
    x = rng.standard_normal((2, 4))  # batch_size=2, input_features=4
    grad_output = rng.standard_normal((2, 3))  # batch_size=2, output_features=3
    # Enter training mode
    with net.training_mode():
        linear.forward(x)  # Necessary to cache input in `x` for gradient computation
        grad_input = linear.backward(grad_output)
    assert grad_input.shape == (2, 4), "Gradient input shape should match input shape"

def test_linear_parameter_update(linear_small):  # noqa
    linear = linear_small
    rng = np.random.default_rng()
    x = rng.standard_normal((1, 4))
    grad_output = rng.standard_normal((1, 3))
    optimizer = net.SGD(learning_rate=0.01)
    initial_weights = np.copy(linear.weights)
    initial_biases = np.copy(linear.biases)
    with net.training_mode():
        linear.forward(x)
        linear.backward(grad_output)
        weight_grad = np.copy(linear.weight_grad)
        bias_grad = np.copy(linear.bias_grad)
        linear.step(optimizer)
    assert np.array_equal(linear.weights, initial_weights - (0.01 * weight_grad))
    assert np.array_equal(linear.biases, initial_biases - (0.01 * bias_grad))

def test_linear_very_large_numbers(linear_small):  # noqa
    linear = linear_small
    x = np.full((1, 4), 1e150)
    output = linear(x)
    assert not np.isnan(output).any()

def test_linear_very_small_numbers(linear_small):  # noqa
    linear = linear_small
    x = np.full((1, 4), 1e-150)
    output = linear(x)
    assert not np.isnan(output).any()

def test_linear_gradient_using_numerical_approximation(linear_small):  # noqa
    """
    Test to perform numerical gradient checking on a Linear layer.
    Gradient checking is a technique to approximate the gradient computed by backpropagation
    and compare it with the gradient calculated using a numerical approximation method.
    This test ensures the backpropagation implementation is correct.
    """
    # Initialize the linear layer and random number generator
    linear = linear_small
    rng = np.random.default_rng()
    x = rng.standard_normal((3, 4))  # single data point with 4 features
    eps = 1e-4  # a small number to move the weights

    # Compute the original output of the layer to be used in gradient computation
    original_output = linear(x)
    # Enter training mode
    with net.training_mode():
        for i in range(linear.weights.shape[0]):  # loop over rows of the weights matrix
            for j in range(linear.weights.shape[1]):  # loop over columns
                old_value = linear.weights[i, j]
                # Increase the current weight by a small value and compute the output
                linear.weights[i, j] = old_value + eps
                output_plus = linear(x)
                # Decrease the current weight by a small value and compute the output
                linear.weights[i, j] = old_value - eps
                output_minus = linear(x)
                # Compute the numerical gradient
                estimated_gradient = (output_plus - output_minus).sum() / (2 * eps)
                # Reset the weight to its original value
                linear.weights[i, j] = old_value
                # Compute the gradient via backpropagation
                grad_output = np.ones_like(original_output)
                linear.backward(grad_output)
                real_gradient = linear.weight_grad[i, j]
                # Verify that the computed gradient matches the numerical estimate
                assert estimated_gradient == pytest.approx(real_gradient, abs=1e-5)

def test_relu_forward_positive():   #noqa
    """Test the ReLU forward function with positive and zero inputs."""
    inputs = np.array([[1.0, -1.0, 0.0], [2.0, -2.0, 0.0]])
    expected_outputs = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    actual_outputs = net.ReLU()(inputs)
    np.testing.assert_array_equal(
        actual_outputs, expected_outputs,
        err_msg="ReLU forward failed for mixed inputs",
    )

def test_relu_forward_all_negative():   #noqa
    """Test the ReLU forward function with all negative inputs."""
    inputs = np.array([[-1.0, -0.5, -2.0]])
    expected_outputs = np.zeros_like(inputs)
    actual_outputs = net.ReLU()(inputs)
    np.testing.assert_array_equal(
        actual_outputs, expected_outputs,
        err_msg="ReLU forward failed for all negative inputs",
    )

def test_relu_backward():  # noqa
    """
    Test the ReLU backward function.

    This test verifies that the gradient passed through ReLU during the backward pass is
    correctly gated by the activation from the forward pass. The ReLU derivative outputs 1
    where the input is positive and 0 otherwise, thus the backward pass should propagate
    gradients only through those elements that had a positive input in the forward pass.
    """
    # Input array for testing, which includes positive, negative, and zero values.
    inputs = np.array([
        [2.0, -3.0, 0.0],
        [1.5, 0.0, -2.0],
    ])
    relu = net.ReLU()
    with net.training_mode():
        relu.forward(inputs)
        # Gradients received from the next layer in the network (or from the loss function).
        # These are hypothetical values to test the gating behavior of ReLU.
        grad_output = np.array([
            [5.0, 2.0, 3.0],
            [4.0, 1.0, 2.0],
        ])
        actual_grad = relu.backward(grad_output)
    # ReLU derivative is 1 for positive inputs and 0 for non-positive inputs (negative or zero).
    # Thus, gradients should only pass through where the inputs were positive.
    # For [2.0, -3.0, 0.0], the gradient [5.0, 2.0, 3.0] should be gated to [5.0, 0.0, 0.0].
    # For [1.5, 0.0, -2.0], the gradient [4.0, 1.0, 2.0] should be gated to [4.0, 0.0, 0.0].
    expected_grad = np.array([
        [5.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])
    # Check if the computed gradients match the expected values.
    np.testing.assert_array_equal(actual_grad, expected_grad, err_msg="ReLU backward failed")

def test_relu_no_training_mode():   #noqa
    """Test that the ReLU forward and backward functions handle non-training mode."""
    inputs = np.array([[1.0, -1.0, 0.0], [2.0, -2.0, 0.0]])
    relu = net.ReLU()
    relu.forward(inputs)  # This should not cache output

    grad_output = np.array([[1.0, 2.0, 3.0], [0.5, 0.0, 1.0]])
    relu.output = 1  # Mock output to ensure error is due to training mode
    with pytest.raises(AssertionError):
        relu.backward(grad_output)

@pytest.mark.parametrize(("input_array", "expected_output"), [
    (np.array([[1, -1, 0], [2, -2, 0]]), np.array([[1, 0, 0], [2, 0, 0]])),
    (np.array([[-1, -2, -3], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 0]])),
])
def test_relu_forward_varied_inputs(input_array, expected_output):   #noqa
    """Test the ReLU forward function with varied inputs using parametrization."""
    output = net.ReLU()(input_array)
    np.testing.assert_array_equal(output, expected_output)

def test_softmax_forward_normal():  # noqa
    """Test the softmax forward function with normal values."""
    inputs = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])
    expected_outputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    expected_outputs /= expected_outputs.sum(axis=1, keepdims=True)
    outputs = net.Softmax()(inputs)
    np.testing.assert_array_almost_equal(outputs, expected_outputs)

def test_softmax_forward_large_values():  # noqa
    """Test softmax with large values to check for stability."""
    inputs = np.array([[1000, 1001, 1002],
                       [1000, 1001, 1002]])
    outputs = net.Softmax()(inputs)
    expected_outputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    expected_outputs /= expected_outputs.sum(axis=1, keepdims=True)

    np.testing.assert_array_almost_equal(outputs, expected_outputs)

def test_softmax_forward_identical_values():  # noqa
    """Test softmax with identical values."""
    inputs = np.array([[1000, 1000, 1000],
                       [1000, 1000, 1000]])
    outputs = net.Softmax()(inputs)
    expected_outputs = np.ones(inputs.shape) / inputs.shape[1]

    np.testing.assert_array_almost_equal(outputs, expected_outputs)

def test_softmax_backward():  # noqa
    """Test the softmax backward function."""
    logits = np.array([[1.0, 2.0, 3.0]])
    softmax = net.Softmax()
    with net.training_mode():
        predictions = softmax.forward(logits)
        grad_output = np.array([[0.1, 0.3, 0.6]])
        grad = softmax.backward(grad_output)

    # Calculate the expected gradient using the formula derived for softmax derivatives
    # ∂L/∂z_j = S_j * ( ∂L/∂S_j - sum(S_i * ∂L/∂S_i) )
    s = predictions
    expected_grad = np.zeros_like(s)
    for i in range(len(s)):
        si = s[i]
        gi = grad_output[i]
        dL_dsi = si * (gi - np.dot(si, gi))  # noqa
        expected_grad[i] = dL_dsi

    np.testing.assert_array_almost_equal(grad, expected_grad)

def test_cross_entropy_loss_correctness():  # noqa
    """Test to ensure the cross-entropy loss is calculated correctly from logits."""
    loss_func = net.CrossEntropyLoss()

    # Test case 1: Standard case with varying correct class indices
    # Using logits instead of probabilities
    logits = np.array([
        [0.1, 1.2, 0.3],  # Assume these are logits and not probabilities
        [2.0, -1.0, 0.0],
        [-0.5, 0.0, 1.5],
    ])
    targets = np.array([1, 0, 2])
    # Computing softmax probabilities manually for the expected loss computation
    softmax_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    expected_losses = -np.log(softmax_probs[np.arange(len(targets)), targets])
    expected_loss = np.mean(expected_losses)

    actual_loss = loss_func(logits, targets)
    assert actual_loss == pytest.approx(expected_loss, abs=1e-5), \
        "CrossEntropyLoss mismatch in logits case"

def test_training_mode_caching():  # noqa
    """
    Test that CrossEntropyLoss only caches logits and targets when Module.training is True,
    and verify that the context manager properly sets and resets the training mode.
    """
    logits = np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]])
    targets = np.array([0, 1])

    assert net.Module.training is False, "Module should not be in training mode initially"
    # Test with the context manager for training mode
    with net.training_mode():
        assert net.Module.training is True, "Module should be in training mode within context"
        loss_func = net.CrossEntropyLoss()
        assert loss_func.logits is None, "Should not have any logits initially"
        assert loss_func.targets is None, "Should not have any targets initially"
        loss_func(logits, targets)
        assert loss_func.logits is not None, "Should cache logits in training mode"
        assert (loss_func.logits == logits).all(), "Cached logits should match input"
        assert loss_func.targets is not None, "Should cache targets in training mode"
        assert (loss_func.targets == targets).all(), "Cached targets should match input"
        loss_func.backward()
        assert loss_func.logits is None, "Should clear logits after backward pass"
        assert loss_func.targets is None, "Should clear targets after backward pass"

    assert net.Module.training is False, "Module should not be in training mode outside context"
    loss_func(logits, targets)
    assert loss_func.logits is None, "Should not cache logits outside training mode"
    assert loss_func.targets is None, "Should not cache targets outside training mode"
    # calling backward outside training mode should raise an error
    loss_func.logits = logits  # mock logits to ensure error is due to training mode
    loss_func.targets = targets  # mock targets to ensure error is due to training mode
    pytest.raises(AssertionError, loss_func.backward)

def test_loss_computation_independence_from_mode():  # noqa
    """Test that the computation of the loss does not depend on the training mode."""
    logits = np.array([[0.2, 0.8], [0.5, 0.5]])
    targets = np.array([1, 0])

    with net.training_mode():
        loss_module = net.CrossEntropyLoss()
        loss_training = loss_module(logits, targets)

    loss_module = net.CrossEntropyLoss()
    loss_inference = loss_module(logits, targets)

    # Verify that loss values are identical regardless of the mode
    np.testing.assert_almost_equal(
        loss_training, loss_inference,
        decimal=5, err_msg="Loss computation should be consistent across modes",
    )

def test_cross_entropy_gradient_correctness():  # noqa
    """Test to ensure the gradients are computed correctly for the cross-entropy loss."""
    # Mock data: single sample, three classes
    logits = np.array([[0.2, 0.5, 0.3]])
    targets = np.array([1])  # correct class index

    loss_module = net.CrossEntropyLoss()

    with net.training_mode():
        loss_module(logits, targets)
        actual_grad = loss_module.backward()

    # The gradient of the cross-entropy loss with respect to the input to the softmax is p-y,
    # due to the derivative of cross-entropy and softmax combined and the corresponding
    # simplification.
    softmax_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    diagnol = np.eye(3)
    expected_grad = softmax_probs - diagnol[targets]

    # Validate gradients
    np.testing.assert_array_almost_equal(
        actual_grad, expected_grad,
        decimal=5, err_msg="Gradient of CrossEntropyLoss is incorrect",
    )

def test_initialization():  # noqa
    layer = net.BatchNorm(10)
    assert layer.gamma.shape == (1, 10)
    assert np.allclose(layer.gamma, 1, atol=0.05)
    assert layer.beta.shape == (1, 10)
    assert np.allclose(layer.beta, 0, atol=0.05)
    assert layer.running_mean.shape == (1, 10)
    assert np.all(layer.running_mean == 0)
    assert layer.running_var.shape == (1, 10)
    assert np.all(layer.running_var == 1)

def test_forward_training_mode():  # noqa
    layer = net.BatchNorm(10)
    rng = np.random.default_rng()
    x = rng.standard_normal((100, 10))
    with net.training_mode():
        output = layer(x)
    assert output.shape == x.shape
    # Check that the output mean is approximately zero and std is close to one because
    # the layer is initialized to normalize the input data and we have not updated the parameters
    assert np.allclose(output.mean(axis=0), 0, atol=0.05)
    assert np.allclose(output.std(axis=0), 1, atol=0.05)

def test_forward_inference_mode():  # noqa
    layer = net.BatchNorm(10)
    rng = np.random.default_rng()
    x = rng.standard_normal((100, 10))
    output = layer(x)
    assert output.shape == x.shape
    # we have not ran training mode, so the output should be close to the input
    assert np.allclose(output, x, atol=0.2)

def test_backward_pass():  # noqa
    layer = net.BatchNorm(10)
    rng = np.random.default_rng()
    x = rng.standard_normal((100, 10))
    grad_output = rng.standard_normal((100, 10))
    with net.training_mode():
        layer(x)  # Forward pass to set internal states
        grad_input = layer.backward(grad_output)
    assert grad_input.shape == x.shape

def test_parameter_updates():  # noqa
    layer = net.BatchNorm(10)
    optimizer = net.SGD(learning_rate=0.01)
    rng = np.random.default_rng()
    x = rng.standard_normal((100, 10))
    grad_output = rng.standard_normal((100, 10))
    with net.training_mode():
        _ = layer.forward(x)
        layer.backward(grad_output)
        old_gamma = layer.gamma.copy()
        old_beta = layer.beta.copy()
        layer.step(optimizer)
    # Check that parameters have been updated
    assert not np.array_equal(layer.gamma, old_gamma)
    assert not np.array_equal(layer.beta, old_beta)

def numerical_gradient(layer, x, grad_output, epsilon=1e-5):  # noqa
    """Utility function to compute the numerical gradients of Batch Norm's parameters."""
    grad_approx = np.zeros_like(layer.gamma)
    for i in range(layer.gamma.size):
        # Perturb gamma positively
        layer.gamma.flat[i] += epsilon
        out1 = layer.forward(x)
        loss1 = np.sum(out1 * grad_output)
        # Perturb gamma negatively
        layer.gamma.flat[i] -= 2 * epsilon
        out2 = layer.forward(x)
        loss2 = np.sum(out2 * grad_output)
        # Compute the approximate gradient
        grad_approx.flat[i] = (loss1 - loss2) / (2 * epsilon)
        # Reset gamma to original value
        layer.gamma.flat[i] += epsilon
    return grad_approx

def test_batch_norm_backward_gradients():  # noqa
    layer = net.BatchNorm(num_features=10)
    rng = np.random.default_rng(seed=42)
    x = rng.standard_normal((10, 10))
    grad_output = rng.standard_normal((10, 10))

    # Perform forward pass to set internal states needed for backward pass
    with net.training_mode():
        _ = layer.forward(x)
        layer.backward(grad_output)
        num_grad_gamma = numerical_gradient(layer, x, grad_output)

    # Check if the analytical gradients are close to the numerical ones
    assert np.allclose(layer.gamma_grad, num_grad_gamma, atol=1e-5), \
        f"Analytical gamma gradients do not match numerical gradients. " \
        f"Analytical: {layer.gamma_grad}, Numerical: {num_grad_gamma}"

def test_Conv2D_initialization():  # noqa
    layer = net.Conv2D(3, 2, 5, stride=1, padding=0, seed=42)
    assert layer.weights.shape == (2, 3, 5, 5), "Weights initialized with incorrect shape"
    assert layer.biases.shape == (2,), "Biases initialized with incorrect shape"
    assert np.allclose(layer.biases, np.zeros(2)), "Biases should be initialized to zero"

def test_Conv2D_forward_pass():  # noqa
    rng = np.random.default_rng(seed=42)
    expected_shape = (1, 1, 4, 4)
    input_tensor = rng.normal(loc=0, scale=1, size=expected_shape)
    layer = net.Conv2D(1, 1, 3, stride=1, padding=1, seed=42)
    output = layer.forward(input_tensor)
    assert output.shape == expected_shape, "Forward pass output shape is incorrect"

def conv_2d_numerical_grad_check(layer, input_tensor, grad_output, epsilon=1e-5):  # noqa
    """
    This function performs numerical gradient checking for the Conv2D layer. It perturbs the input
    tensor by a small epsilon value and computes the approximate gradient of the loss with respect
    to the input tensor. It then compares this numerical gradient with the analytical gradient
    computed by the layer's backward method.
    """  # noqa: D404
    approx_grad = np.zeros_like(input_tensor)
    # Iterate over each element of the input tensor
    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[1]):
            for k in range(input_tensor.shape[2]):
                for l in range(input_tensor.shape[3]):  # noqa
                    # Perturb input tensor
                    old_val = input_tensor[i, j, k, l]
                    input_tensor[i, j, k, l] = old_val + epsilon
                    plus_loss = np.sum(layer.forward(input_tensor) * grad_output)
                    input_tensor[i, j, k, l] = old_val - epsilon
                    minus_loss = np.sum(layer.forward(input_tensor) * grad_output)
                    # Approximate gradient
                    approx_grad[i, j, k, l] = (plus_loss - minus_loss) / (2 * epsilon)
                    input_tensor[i, j, k, l] = old_val  # Reset the value
    assert np.allclose(approx_grad, layer.backward(grad_output), atol=1e-4), \
        "Backward pass gradient check failed"

def test_Conv2D_backward_pass_numerically():  # noqa
    """This test performs numerical gradient checking for the Conv2D layer."""  # noqa: D404
    rng = np.random.default_rng(seed=42)
    input_tensor = rng.normal(loc=0, scale=1, size=(1, 1, 4, 4))
    layer = net.Conv2D(1, 1, 3, stride=1, padding=1, seed=42)
    with net.training_mode():
        output = layer.forward(input_tensor)
        # Grad output mimicking the next layer's gradient
        grad_output = rng.normal(loc=0, scale=1, size=output.shape)
        conv_2d_numerical_grad_check(layer, input_tensor, grad_output)

def test_Conv2D_parameter_update():  # noqa
    rng = np.random.default_rng(seed=42)
    input_tensor = rng.normal(loc=0, scale=1, size=(1, 1, 4, 4))
    optimizer = net.SGD(learning_rate=0.01)
    layer = net.Conv2D(1, 1, 3, stride=1, padding=0, seed=42)
    grad_output = rng.normal(loc=0, scale=1, size=(1, 1, 2, 2))  # Reduced size due to no padding
    with net.training_mode():
        _ = layer.forward(input_tensor)
        _ = layer.backward(grad_output)
        old_weights = np.copy(layer.weights)
        old_biases = np.copy(layer.biases)
        layer._step(optimizer)
    assert not np.array_equal(old_weights, layer.weights), "Weights should have been updated"
    assert not np.array_equal(old_biases, layer.biases), "Biases should have been updated"

def test_MaxPool2D_forward_pass():  # noqa
    layer = net.MaxPool2D(kernel_size=2, stride=2)
    input_tensor = np.array([
        [[1, 3, 2, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]],
    ]).reshape(1, 1, 4, 4)  # Batch size 1, 1 channel, 4x4 input

    output = layer.forward(input_tensor)
    expected_output = np.array([
        [[6, 8],
         [14, 16]],
    ]).reshape(1, 1, 2, 2)  # 2x2 output due to 2x2 kernel and stride 2

    assert output.shape == (1, 1, 2, 2), "Output shape is incorrect"
    assert np.array_equal(output, expected_output), "Max pooling result is incorrect"

def max_2d_numerical_gradient_check(layer, input_tensor, grad_output, epsilon=1e-6):  # noqa
    """
    This function performs numerical gradient checking for the MaxPool2D layer. It perturbs the
    input tensor by a small epsilon value and computes the approximate gradient of the loss with
    respect to the input tensor. It then compares this numerical gradient with the analytical
    gradient computed by the layer's backward method.
    """  # noqa: D404
    num_grad = np.zeros_like(input_tensor)
    it = np.nditer(input_tensor, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = input_tensor[ix]
        # Increment by epsilon
        input_tensor[ix] = old_value + epsilon
        pos_output = layer.forward(input_tensor)
        pos_loss = np.sum(pos_output * grad_output)
        # Decrement by epsilon
        input_tensor[ix] = old_value - epsilon
        neg_output = layer.forward(input_tensor)
        neg_loss = np.sum(neg_output * grad_output)
        # Compute numerical gradient
        num_grad[ix] = (pos_loss - neg_loss) / (2 * epsilon)
        # Restore original value
        input_tensor[ix] = old_value
        it.iternext()
    return num_grad

def test_MaxPool2D_backward_pass_numerically():  # noqa
    """This test performs numerical gradient checking for the MaxPool2D layer."""  # noqa
    layer = net.MaxPool2D(kernel_size=2, stride=2)
    rng = np.random.default_rng(seed=42)
    input_tensor = rng.normal(loc=0, scale=1, size=(1, 1, 4, 4))
    with net.training_mode():
        output = layer.forward(input_tensor)
        grad_output = rng.normal(loc=0, scale=1, size=output.shape)
        # Calculate analytical backward pass
        analytic_grad = layer.backward(grad_output)
        # Calculate numerical gradients
        num_grad = max_2d_numerical_gradient_check(layer, input_tensor, grad_output)
    # Assert the closeness of analytical and numerical gradients
    assert np.allclose(num_grad, analytic_grad, atol=1e-5), \
        "Numerical and analytical gradient mismatch"

def test_Flatten_forward_pass():  # noqa
    rng = np.random.default_rng(seed=42)
    input_tensor = rng.normal(loc=0, scale=1, size=(2, 3, 4, 5))
    flatten = net.Flatten()
    output = flatten.forward(input_tensor)
    # Expected flattened shape (batch_size, product of other dimensions)
    expected_shape = (2, 3 * 4 * 5)
    assert output.shape == expected_shape, \
        "Flatten forward pass did not produce the correct shape."

def test_Flatten_backward_pass():  # noqa
    rng = np.random.default_rng(seed=42)
    input_tensor = rng.normal(loc=0, scale=1, size=(2, 3, 4, 5))
    flatten = net.Flatten()
    with net.training_mode():
        _ = flatten.forward(input_tensor)
        grad_output = rng.normal(loc=0, scale=1, size=(2, 3 * 4 * 5))
        grad_input = flatten.backward(grad_output)
    assert grad_input.shape == input_tensor.shape, \
        "Flatten backward pass did not restore the gradient to the correct shape."
