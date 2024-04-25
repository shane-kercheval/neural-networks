#include "linear.h"

namespace torchcpp {

Linear::Linear(int in_features, int out_features, double (*weight_init_scale)(int, int), unsigned int seed)
    : in_features_(in_features),
      out_features_(out_features),
      weight_grad_(MatrixXd::Zero(in_features, out_features)),
      bias_grad_(MatrixXd::Zero(1, out_features)) {

    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(
        0.0,
        // if weight_init_scale is a nullptr, use he initialization; otherwise, use the provided function
        weight_init_scale ? weight_init_scale(in_features, out_features) : he_init_scale(in_features, out_features)
    );

    // The `unaryExpr` function is a method in Eigen for element-wise operations on matrices.
    // It applies a given unary function (a function that takes a single argument) to each element
    // of the matrix or array.
    // `[&]` captures all local variables by reference, so distribution and generator are accessible
    // inside the lambda. `(double x)` is the argument to the lambda function
    weights_ = MatrixXd(in_features, out_features).
        unaryExpr([&](double x) { return distribution(generator); });
    biases_ = MatrixXd::Zero(1, out_features);
}

MatrixXd Linear::forward(const MatrixXd& x) {
    if (x.cols() != in_features_) {
        throw std::runtime_error("Input size does not match the number of input features.");
    }
    if (Module::training) {
        x_cached_ = x;  // this performs a deep copy
    }
    return (x * weights_).rowwise() + biases_.row(0);
}

MatrixXd Linear::backward_impl(const MatrixXd& grad_output) {
    weight_grad_ = x_cached_.transpose() * grad_output;
    bias_grad_ = grad_output.colwise().sum();
    return grad_output * weights_.transpose();
}

void Linear::step_imp(const std::function<void(MatrixXd&, MatrixXd&)>& optimizer) {
    optimizer(weights_, weight_grad_);
    optimizer(biases_, bias_grad_);
}

void Linear::zero_grad() {
    weight_grad_.setZero();
    bias_grad_.setZero();
}

}
