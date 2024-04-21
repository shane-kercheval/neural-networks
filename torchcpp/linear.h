// #pragma once
// #include "Module.h"

// class Linear : public Module {
// private:
//     Eigen::MatrixXd weights, biases;
//     Eigen::MatrixXd weight_grad, bias_grad;
//     Eigen::MatrixXd input_cache;  // for backward computation

// public:
//     Linear(int in_features, int out_features);
//     virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) override;
//     virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
//     virtual void step() override;
// };
