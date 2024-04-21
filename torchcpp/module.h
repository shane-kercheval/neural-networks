// #pragma once
// #include <Eigen/Dense>

// class Module {
// public:
//     static bool training;

//     virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) = 0;
//     virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) = 0;
//     virtual void step() = 0;
//     virtual ~Module() {} // Virtual destructor to ensure proper cleanup of derived classes
// };

// bool Module::training = false;


// /**
//     // Training loop
//     TrainingMode training_mode;
//     training_mode.enter();
//     // Forward, backward, and step operations
//     training_mode.exit();
// */
// class TrainingMode {
// public:
//     void enter() { Module::training = true; }
//     void exit() { Module::training = false; }
// };
