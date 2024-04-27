#include "loss.h"
#include "utils.h"

namespace torchcpp {

double CrossEntropyLoss::forward(const MatrixXd& logits, const VectorXi& targets) {
    if (Module::training) {
        logits_cache_ = logits; // Cache logits for backward pass
        targets_cache_ = targets; // Cache targets for backward pass
    }
    Eigen::MatrixXd softmax_probs = softmax(logits);

    const int num_examples = logits.rows();
    double loss = 0.0;
    for (int i = 0; i < num_examples; ++i) {
        double prob = softmax_probs(i, targets(i));
        loss += -std::log(std::max(prob, 1e-15));
    }
    return loss / num_examples;
}

Eigen::MatrixXd CrossEntropyLoss::backward() {
    if (!Module::training) {
        throw std::logic_error("Module::backward should only be called during training");
    }
    const int num_examples = logits_cache_.rows();
    if (num_examples == 0) {
        throw std::logic_error("No data in cache. Make sure to call forward first.");
    }
    Eigen::MatrixXd softmax_probs = softmax(logits_cache_);
    Eigen::MatrixXd grad_logits = softmax_probs;
    for (int i = 0; i < targets_cache_.size(); ++i) {
        grad_logits(i, targets_cache_(i)) -= 1;
    }
    grad_logits /= targets_cache_.size();
    return grad_logits;
}

SGD::SGD(double learning_rate) : learning_rate_(learning_rate) {}

void SGD::operator()(Eigen::MatrixXd& parameters, const Eigen::MatrixXd& gradients) const {
    parameters -= learning_rate_ * gradients;
}

}
