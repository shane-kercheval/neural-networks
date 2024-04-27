#pragma once
#include <Eigen/Dense>
#include "module.h"

using Eigen::MatrixXd;

namespace torchcpp {

class ReLU : public Module {
public:
    ReLU() = default;

    /**
     * @brief Forward pass of the ReLU activation function.
     * 
     * @param x Input data.
    */
    MatrixXd forward(const MatrixXd& x) override;

protected:
    /**
     * @brief Backward pass of ReLU.
     * 
     * The ReLU function is defined as:
     *      `f(x) = max(0, x)`
     * 
     * The derivative of ReLU:
     *  `f'(x) = 1 if x > 0 else 0`
     * 
     * Therefore:
     *      `∂L/∂x = ∂L/∂f * ∂f/∂x`, where
     *          - `∂f/∂x` = f'(x)
     *          - `∂L/∂f` is the gradient of the loss function with respect to the output of this
     *             layer, which is passed as grad_output.
     * 
     * @param grad_output (∂L/∂y) Gradient of the loss with respect to the output of this layer.
     * Calculated in the next layer and passed to this layer during backpropagation.
    */
    MatrixXd backward_impl(const MatrixXd& grad_output) override;

private:
    MatrixXd output_cache_;  // cache the input for the backward pass (during training only)
};

}
