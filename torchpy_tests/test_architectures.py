"""Tests for the torchpy.architectures module."""
from numpy.random import default_rng
from torchpy import architectures as arch
from torchpy import net


def test__MLP():  # noqa
    mlp = arch.MLP(in_features=10, out_features=2, hidden_features=[16, 32])
    assert mlp is not None
    assert len(mlp.layers) == 7
    assert isinstance(mlp.layers[0], net.Linear)
    assert mlp.layers[0].weights.shape == (10, 16)
    assert isinstance(mlp.layers[3], net.Linear)
    assert mlp.layers[3].weights.shape == (16, 32)
    assert isinstance(mlp.layers[6], net.Linear)
    assert mlp.layers[6].weights.shape == (32, 2)
    rng = default_rng(42)
    x = rng.normal(loc=0, scale=1, size=(100, 10))
    y = mlp(x)
    assert y.shape == (100, 2)
    # test backward pass (gradients tested in individual layers)
    with net.training_mode():
        y = mlp(x)
        assert y.shape == (100, 2)
        grad_output = rng.normal(loc=0, scale=1, size=(100, 2))
        mlp.backward(grad_output)
        mlp.step(optimizer=lambda p, g: p - 0.1 * g)
    assert str(mlp)
