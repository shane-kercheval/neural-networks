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
    */
    MatrixXd forward(const MatrixXd& x) override;

protected:
    /**
     * @brief Backward pass of the ReLU activation function.
    */
    MatrixXd backward_impl(const MatrixXd& grad_output) override;

private:
    MatrixXd output_cache_;  // cache the input for the backward pass (during training only)
};

}
