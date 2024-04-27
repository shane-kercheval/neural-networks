#pragma once
#include <Eigen/Dense>
#include "module.h"

using Eigen::MatrixXd;
using Eigen::VectorXi;

namespace torchcpp {

/**
 * @brief Combines Softmax and Cross-Entropy loss into one single class for stability and
 * efficiency.
*/
class CrossEntropyLoss {

public:
    CrossEntropyLoss() = default;
    ~CrossEntropyLoss() = default;
    // for now, we will disable copy and move operations for all modules; i'd like to
    // understand when they would be used before enabling them.
    // copy constructor
    CrossEntropyLoss(const CrossEntropyLoss& other) = delete;
    // copy assignment operator
    CrossEntropyLoss& operator=(const CrossEntropyLoss& other) = delete;
    // move constructor
    // by marking these functions as noexcept, we're indicating that they won't throw exceptions.
    // This is important for maintaining strong exception safety (i.e., the program remains in
    // a valid state and no resources are leaked if an exception is thrown).
    CrossEntropyLoss(CrossEntropyLoss&& other) noexcept = delete;
    // move assignment operator
    CrossEntropyLoss& operator=(CrossEntropyLoss&& other) noexcept = delete;

    /**
     * @brief Computes the cross-entropy loss from logits and targets directly.
     * 
     * @param logits Logits array (before softmax).
     * @param targets Array of target class indices. The `i` means that the elements of the vector
     * are integers.
    */
    double forward(const MatrixXd& logits, const VectorXi& targets);

    /**
     * @brief Computes and returns the gradient of the loss with respect to the logits.
     * 
     * @param grad_output (∂L/∂p) Gradient of the loss with respect to the output of this layer.
     * Calculated in the next layer and passed to this layer during backpropagation.
    */
    MatrixXd backward();

protected:
    MatrixXd logits_cache_;
    VectorXi targets_cache_;
};

}
