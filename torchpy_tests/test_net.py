"""Tests the neural network modules in torchpy.net."""
import numpy as np
import pytest
from torchpy import net


@pytest.fixture()
def softmax():  # noqa
    net.Module.training = True
    return net.Softmax()


def test_softmax_forward_normal(softmax):  # noqa
    """Test the softmax forward function with normal values."""
    inputs = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])
    expected_outputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    expected_outputs /= expected_outputs.sum(axis=1, keepdims=True)
    outputs = softmax.forward(inputs)
    np.testing.assert_array_almost_equal(outputs, expected_outputs)

def test_softmax_forward_large_values(softmax):  # noqa
    """Test softmax with large values to check for stability."""
    inputs = np.array([[1000, 1001, 1002],
                       [1000, 1001, 1002]])
    outputs = softmax.forward(inputs)
    expected_outputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    expected_outputs /= expected_outputs.sum(axis=1, keepdims=True)

    np.testing.assert_array_almost_equal(outputs, expected_outputs)

def test_softmax_forward_identical_values(softmax):  # noqa
    """Test softmax with identical values."""
    inputs = np.array([[1000, 1000, 1000],
                       [1000, 1000, 1000]])
    outputs = softmax.forward(inputs)
    expected_outputs = np.ones(inputs.shape) / inputs.shape[1]

    np.testing.assert_array_almost_equal(outputs, expected_outputs)

def test_softmax_backward(softmax):  # noqa
    """Test the softmax backward function."""
    logits = np.array([[1.0, 2.0, 3.0]])
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
    """Test to ensure the cross-entropy loss is calculated correctly."""
    loss_func = net.CrossEntropyLoss()

    # Test case 1: Standard case with varying correct class indices
    predictions = np.array([
        [0.1, 0.6, 0.3],
        [0.8, 0.1, 0.1],
        [0.3, 0.1, 0.6],
    ])
    # 1st sample: correct class index 1
    # 2nd sample: correct class index 0
    # 3rd sample: correct class index 2
    targets = np.array([1, 0, 2])
    expected_losses = -np.log(np.array([0.6, 0.8, 0.6]))
    expected_loss = np.mean(expected_losses)

    actual_loss = loss_func(predictions, targets)
    assert actual_loss == pytest.approx(expected_loss, abs=1e-5), \
        "CrossEntropyLoss mismatch in edge cases"

    # Test case 2: Edge case where predictions include 0 and 1
    inputs_edge = np.array([
        [0, 1, 0],  # probabilities should be clipped
        [0, 0, 1],  # probabilities should be clipped
        [1, 0, 0],   # probabilities should be clipped
        [0, 0, 1],  # probabilities should be clipped

    ])
    # correct class indices
    targets_edge = np.array([
        1,  # perfect prediction
        2,  # perfect prediction
        1,  # prediction is 0 for correct class
        0,  # prediction is 0 for correct class
    ])
    # Clipping values to avoid log(0) which leads to -inf
    clipped_probs = np.clip(inputs_edge, 1e-15, 1-1e-15)
    expected_losses_edge = -np.log(clipped_probs[np.arange(len(targets_edge)), targets_edge])
    expected_loss = np.mean(expected_losses_edge)

    actual_loss = loss_func(inputs_edge, targets_edge)
    assert actual_loss == pytest.approx(expected_loss, abs=1e-5), \
        "CrossEntropyLoss mismatch in edge cases"

def test_training_mode_caching():  # noqa
    """
    Test that CrossEntropyLoss only caches predictions and targets when Module.training is True,
    and verify that the context manager properly sets and resets the training mode.
    """
    predictions = np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]])
    targets = np.array([0, 1])

    assert net.Module.training is False, "Module should not be in training mode initially"
    # Test with the context manager for training mode
    with net.Module.training_mode():
        assert net.Module.training is True, "Module should be in training mode within context"
        loss_func = net.CrossEntropyLoss()
        assert loss_func.predictions is None, "Should not have any predictions initially"
        assert loss_func.targets is None, "Should not have any targets initially"
        loss_func(predictions, targets)
        assert loss_func.predictions is not None, "Should cache predictions in training mode"

        assert (loss_func.predictions == predictions).all(), \
            "Cached predictions should match input"
        assert (loss_func.targets == targets).all(), "Cached predictions should match input"

        assert loss_func.targets is not None, "Should cache targets in training mode"
        loss_func.backward()
        assert loss_func.predictions is None, "Should clear predictions after backward pass"
        assert loss_func.targets is None, "Should clear targets after backward pass"


    assert net.Module.training is False, "Module should not be in training mode outside context"
    loss_func(predictions, targets)
    assert loss_func.predictions is None, "Should not cache predictions outside training mode"
    assert loss_func.targets is None, "Should not cache targets outside training mode"
    # calling backward outside training mode should raise an error
    loss_func.predictions = 1  # mock predictions to ensure error is due to training mode
    loss_func.targets = 1  # mock targets to ensure error is due to training mode
    pytest.raises(AssertionError, loss_func.backward)

def test_loss_computation_independence_from_mode():  # noqa
    """Test that the computation of the loss does not depend on the training mode."""
    inputs = np.array([[0.2, 0.8], [0.5, 0.5]])
    targets = np.array([1, 0])

    with net.Module.training_mode():
        loss_module = net.CrossEntropyLoss()
        loss_training = loss_module.forward(inputs, targets)

    # Module.training should automatically revert to False outside of 'with' block
    loss_module = net.CrossEntropyLoss()
    loss_inference = loss_module.forward(inputs, targets)

    # Verify that loss values are identical regardless of the mode
    np.testing.assert_almost_equal(
        loss_training, loss_inference,
        decimal=5, err_msg="Loss computation should be consistent across modes",
    )

def test_cross_entropy_gradient_correctness():  # noqa
    """Test to ensure the gradients are computed correctly for the cross-entropy loss."""
    # Mock data: single sample, three classes
    predictions = np.array([[0.2, 0.5, 0.3]])
    targets = np.array([1])  # correct class index

    loss_module = net.CrossEntropyLoss()

    with net.Module.training_mode():
        loss_module.forward(predictions, targets)
        actual_grad = loss_module.backward()

    # The gradient of the cross-entropy loss with respect to the input to the softmax is p-y,
    # due to the derivative of cross-entropy and softmax combined and the corresponding
    # simplification.
    diagnol = np.eye(3)
    expected_grad = predictions - diagnol[targets]

    # Validate gradients
    np.testing.assert_array_almost_equal(
        actual_grad, expected_grad,
        decimal=5, err_msg="Gradient of CrossEntropyLoss is incorrect",
    )
