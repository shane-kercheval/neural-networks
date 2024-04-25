# neural-networks

- Repository to learn about neural networks.
- `torchpy` folder contains lightweight python classes that mimic pytorch; meant for learning purposes, not production.
- `torchcpp` folder contains lightweight C++ that mimic pytorch; meant for learning purposes, not production.
- `multilayer_perceptron`, `mlp_batch_normalization`, and `cnn` folders contain .ipynb files that run pytorch, torchpy, and numpy versions of MLPs/CNNs on mnist.
- see `Makefile` for general idea of how to run projects, compile code, etc.
- Dockerfile/compose used for reproducibility

## Python Implementation

- `torchpy` and `torchpy_tests` contain python classes for neural network components (e.g. Module, Linear, RelU)

## C++ Implementation

- `torchcpp` and `torchcpp_tests` contain C++ classes for neural network components (e.g. Module, Linear, RelU)


# Attribution

- https://github.com/karpathy/makemore
    - `names.txt` and some MLP code comes from makemore series
    - highly recommend watching [The spelled-out intro to language modeling: building makemore - Andrej Karpathy](https://www.youtube.com/watch?v=PaCmpygFfXo)
- https://github.com/karpathy/micrograd
