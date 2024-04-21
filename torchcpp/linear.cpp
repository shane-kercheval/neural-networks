// #include "Linear.h"
// #include "Utils.h"
// #include <random>

// // Linear::Linear(int in_features, int out_features) {
// //     weights = he_initialization(in_features, out_features);
// //     biases = Eigen::MatrixXd::Zero(1, out_features);
// // }

// // Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd& x) {
// //     input_cache = x; // Save input for backward pass
// //     return (x * weights).rowwise() + biases;
// // }

// // Eigen::MatrixXd Linear::backward(const Eigen::MatrixXd& grad_output) {
// //     weight_grad = input_cache.transpose() * grad_output;
// //     bias_grad = grad_output.colwise().sum();
// //     return grad_output * weights.transpose();
// // }

// // void Linear::step() {
// //     // Implement the optimizer step here
// // }
