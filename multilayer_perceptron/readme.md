# Multilayer Perceptron w/ mnist dataset

- Pytorch MLP training time (via pytorch_mlp_mnist.ipynb)
    - CPU Apple M2 Max: `952.16 seconds` (`15 minutes 52.16 seconds`)
    - Test loss: `0.18956883251667023`
    - Test accuracy: `0.947`
- Numpy MLP training time (via numpy_mlp_mnist.ipynb)
    - CPU Apple M2 Max: `24.10 seconds`
    - Test loss: `0.14705961978974613`
    - Test accuracy: `0.9591428571428572`
- `torchpy` (Python implementation + Numpy) MLP training time (via torchpy_mlp_mnist.ipynb)
    - CPU Apple M2 Max: `25.76 seconds`
    - Test loss: `0.1470596197897461`
    - Test accuracy: `0.959`
- `torchcpp` (C++ implementation + Eigen) MLP training time (via torchcpp_mlp_mnist.cpp)
    - CPU Apple M2 Max: `609.868 seconds`
    - Test loss: `0.126021`
    - Test accuracy: `0.9637`
