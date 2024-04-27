#include <Eigen/Dense>
#include "torchcpp.h"

namespace tests {

/**
 * This class is a mock implementation of the Module class. It is used to test the Module class
 * and its methods.
*/
class MockModule : public torchcpp::Module {
public:
    MockModule() = default;

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) override {return x; }
    Eigen::MatrixXd weights;
    Eigen::MatrixXd weight_grad;
protected:
    Eigen::MatrixXd backward_impl(const Eigen::MatrixXd& grad_output) override { return grad_output; }
};

/**
 * This class is a mock implementation of the Linear class. It is used to test the Linear class
 * and specifically backpropagation by exposing the protected members of the Linear class.
*/
class TestableLinear : public torchcpp::Linear {
public:
    using Linear::Linear; // Inherit Linear's constructor
    // Expose protected members as public for testing
    MatrixXd& get_weights() { return weights_; }
    MatrixXd& get_weight_grad() { return weight_grad_; }
    MatrixXd& get_biases() { return biases_; }
    MatrixXd& get_bias_grad() { return bias_grad_; }
};

}
