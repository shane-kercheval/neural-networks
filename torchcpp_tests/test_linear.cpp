#include <gtest/gtest.h>
#include <functional>
#include <Eigen/Dense>
#include "torchcpp.h"

using namespace torchcpp;
using Eigen::MatrixXd;



/**
 * This test checks that the forward pass of the Linear layer is correct. It does this by
 * initializing the weights/biases to 0, and then shifting them by 1 (via step()).
 * It then passes an input matrix to the forward method, and checks that the output is as expected
 * based on multiplying the input by the weights and adding the biases.
*/
TEST(TestLinear, forward_pass) {

    // define initializer function that returns 0
    auto zero_init = [](int, int) { return 0.0; };
    // define optimizer that shifts the weights by 1
    auto shift_weights = [](MatrixXd& weights, MatrixXd& _) { 
        weights += MatrixXd::Ones(weights.rows(), weights.cols());
    };
    Linear linear(3, 2, zero_init);
    MatrixXd x(2, 3);
    {
        USING_TRAINING_MODE _;
        linear.step(shift_weights);
    }
    // the weights were initialized to 0 via zero_init, and then shifted by 1 via the optimizer
    // the bias is initialized to 0 by default, and shifted by 1 via the optimizer
    x << 1, 2, 3,
         4, 5, 6;
    MatrixXd output = linear.forward(x);
    // std::cout << output << std::endl;
    MatrixXd expected_output(2, 2);
    // 1, 2, 3  mult by 1, 1,
    // 4, 5, 6          1, 1,
    //                  1, 1
    // is 
    //   6, 6
    //   15, 15
    // plus bias of 1 is
    //   7, 7
    //   16, 16
    expected_output << 7, 7,
                       16, 16;
    ASSERT_TRUE(output.isApprox(expected_output));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
