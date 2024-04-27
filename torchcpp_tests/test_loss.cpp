#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "torchcpp.h"

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
