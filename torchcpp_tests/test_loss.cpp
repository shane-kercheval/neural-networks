#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "torchcpp.h"
#include <iostream>

using namespace torchcpp;
using Eigen::MatrixXd;
using Eigen::VectorXi;

TEST(TestCrossEntropyLoss, forward_pass) {
    CrossEntropyLoss loss;
    MatrixXd logits(3, 3);
    VectorXi targets(3);
    logits << 0.1, 1.2, 0.3,
              2.0, -1.0, 0.0,
              -0.5, 0.0, 1.5;
    targets << 1, 0, 2;
    const double expected_loss = 0.3432551271770566;  // Calculated from python unit test
    ASSERT_NEAR(expected_loss, loss.forward(logits, targets), 1e-9);
}

TEST(TestCrossEntropyLoss, backward_pass) {
    // Python code/results
    // logits = np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1], [0.3, 0.2, 0.5]])
    // targets = np.array([0, 1, 2])
    // # gradient returned
    // array([[-0.23685562986857844, 0.13023127775660523, 0.10662435211197324],
    //        [0.08304780030740974, -0.16609560061481946, 0.08304780030740974],
    //        [0.10662435211197323, 0.09647770346475487, -0.2031020555767281]])
    CrossEntropyLoss loss;
    MatrixXd logits(3, 3);
    VectorXi targets(3);
    logits << 0.2, 0.5, 0.3,
              0.1, 0.8, 0.1,
              0.3, 0.2, 0.5;
    targets << 0, 1, 2;
    {
        USING_TRAINING_MODE _;
        double loss_value = loss.forward(logits, targets);
        ASSERT_NEAR(0.9564629208863787, loss_value, 1e-9);
        MatrixXd grad_logits = loss.backward();
        MatrixXd expected_grad_logits(3, 3);
        expected_grad_logits << -0.23685562986857844, 0.13023127775660523, 0.10662435211197324,
                                0.08304780030740974, -0.16609560061481946, 0.08304780030740974,
                                0.10662435211197323, 0.09647770346475487, -0.2031020555767281;
        ASSERT_TRUE(expected_grad_logits.isApprox(grad_logits));
    }
}

TEST(SGD, updating) {
    SGD optimizer(0.1);
    MatrixXd parameters(3, 3);
    MatrixXd gradients(3, 3);
    parameters << 0.2, 0.5, 0.3,
                  0.1, 0.8, 0.1,
                  0.3, 0.2, 0.5;
    gradients << -0.23685562986857844, 0.13023127775660523, 0.10662435211197324,
                 0.08304780030740974, -0.16609560061481946, 0.08304780030740974,
                 0.10662435211197323, 0.09647770346475487, -0.2031020555767281;
    MatrixXd expected_values = parameters - (0.1 * gradients);
    optimizer(parameters, gradients);
    ASSERT_TRUE(expected_values.isApprox(parameters));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
