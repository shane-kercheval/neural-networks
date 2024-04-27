#pragma once
#include <Eigen/Dense>
#include <random>
#include "torchcpp.h"

using Eigen::MatrixXd;

namespace torchcpp {

const unsigned DEFAULT_SEED = 42;

/** Linear (fully connected) layer. */
class Linear : public Module {
public:

    /**
     * @brief Construct a new Linear object
     * 
     * @param in_features The number of input features.
     * @param out_features The number of output features.
     * @param weight_init_scale The function to use for initializing the weights. If none provided
     * (nullptr), then He initialization is used.
     * @param seed The seed for the random number generator.
    */
    Linear(
        int in_features,
        int out_features,
        double (*weight_init_scale)(int, int) = nullptr,
        unsigned int seed = DEFAULT_SEED
    );

    /**
     * @brief Forward pass of the Linear layer.
     * 
     * @param x Input data of shape (batch_size, in_features).
     * @return Output of the layer of shape (batch_size, out_features).
    */
    MatrixXd forward(const MatrixXd& x) override;

    /**
     * @brief Clear the gradients of the layer.
    */
    void zero_grad() override;

protected:
    /**
     * @brief Returns the gradient with respect to the input.
     * 
     * The Linear layer is defined by: output=input*weights + biases i.e. y=xW+b
     *
     * Where,
     * - `x` is the input matrix (passed to forward) with shape (batch size, in_features)
     * - `W` is the weight matrix with shape (in_features, out_features)
     * - `b` is the bias vector with shape (1, out_features)
     * - `y` is the output matrix with shape (batch size, out_features)
     *
     * Our goal is to compute the gradient of the loss function with respect to all of the
     * parameters of this layer, and to propagate the gradient of the loss function backward
     * to previous layers in the network:
     * 
     * - `∂L/∂W`: The gradient of the loss function with respect to the weights
     * - `∂L/∂b`: gradient of the loss function with respect to the biases
     * - `∂L/∂x`: The gradient of the loss function with respect to the input, which is necessary
     * to propagate the back to previous layers.
     * - `∂L/∂y`: grad_output: the gradient of the loss function with respect to the output y
     * 
     * Gradient of L with respect to the weights is calculated as:
     * 
     * - `∂L/∂W = ∂L/∂y * ∂y/∂W`, where ∂L/∂y is grad_output
     * - `∂y/∂W`: (the partial derivative of y with respect to W) means we treat all other
     * variables (the bias and inputs) as constants) is `x`, because `y = xW + b` and the
     * derivative b is a constant (which is 0) and the derivative of W with respect to W
     * is 1, so the derivative of y with respect to W is x.
     * - so `∂L/∂W = ∂L/∂y * ∂y/∂W` = grad_output * x, but we need to align the dimensions
     * correctly for matrix multiplication, so we transpose x to get the correct shape.
     * The dimension of self.x.T is (in_features, batch_size), and the dimension of
     * grad_output is (batch_size, out_features) so the matrix multiplication is
     * (in_features, batch_size) @ (batch_size, out_features).
     * 
     * Gradient of L with respect to the biases is calculated as:
     * 
     * - `∂L/∂b = ∂L/∂y * ∂y/∂b`, where ∂L/∂y is grad_output
     * - `∂y/∂b  is 1, because `y = xW + b` and W and x are treated as constants, so the\
     * derivative of y with respect to b is simply 1.
     * - So ∂L/∂b = ∂L/∂y * ∂y/∂b = grad_output * 1 = grad_output
     * 
     * Gradient of L with respect to the input (x) is calculated as:
     * 
     * - `∂L/∂x = ∂L/∂y * ∂y/∂x`, where ∂L/∂y is grad_output
     * - `∂y/∂x` is W, because `y = xW + b` where b and W are treated as a constants
     * - so `∂L/∂x = ∂L/∂y * ∂y/∂x = grad_output * W`
     * 
     * @param grad_output: (∂L/∂y) Gradient of the loss with respect to the output of this layer.
     * Calculated in the next layer and passed to this layer during backpropagation.
     * @return Gradient of the loss with respect to the input of this layer.
    */
    MatrixXd backward_impl(const MatrixXd& grad_output) override;

    /**
     * @brief Update the weights and biases of the module using the gradient computed during
     * backpropagation and the optimizer provided.
     * 
     * @param optimizer The optimizer to use for updating the weights and biases.
    */
    void step_imp(const std::function<void(MatrixXd&, MatrixXd&)>& optimizer) override;

protected:
    int in_features_;
    int out_features_;
    MatrixXd weights_;
    // i'm using MatrixXd for biases_ and bias_grad_ because the optimizer function requires
    // a reference to a MatrixXd object, and i can't pass a reference to a VectorXd object
    MatrixXd biases_;
    MatrixXd weight_grad_;
    MatrixXd bias_grad_;
    MatrixXd x_cached_;  // cache the input for the backward pass
};

}
