#include "gtest/gtest.h"
#include "torchcpp.h"
#include <Eigen/Dense>

using namespace torchcpp;
using Eigen::MatrixXd;


/**
 * This test checks that the forward pass of the ReLU activation function is correct.
*/
TEST(ActivationsTest, test_relu_forward) {
    ReLU relu;
    MatrixXd input_1(2, 3);
    input_1 << -1, 0, 2,
         1, -2, 3;
    MatrixXd expected_output_1(2, 3);
    expected_output_1 << 0, 0, 2,
                       1, 0, 3;
    ASSERT_TRUE(expected_output_1.isApprox(relu.forward(input_1)));

    MatrixXd input_2(2, 3);
    input_2 << -0.1, 0, 0.2,
         0.1, -0.2, 0.3;
    MatrixXd expected_output_2(2, 3);
    expected_output_2 << 0, 0, 0.2,
                       0.1, 0, 0.3;
    ASSERT_TRUE(expected_output_2.isApprox(relu.forward(input_2)));
}

/**
 * Test the ReLU backward function. 
 * 
 * This test verifies that the gradient passed through ReLU during the backward pass is correctly
 * gated by the activation from the forward pass. The ReLU derivative outputs 1 where the input is
 * positive and 0 otherwise, thus the backward pass should propagate gradients only through those
 * elements that had a positive input in the forward pass.
*/
TEST(ActivationsTest, test_relu_backward) {
    ReLU relu;
    MatrixXd x(2, 3);
    x << 2.0, -3.0, 0.0,
         1.5, 0.0, -2.0;
    MatrixXd grad_output(2, 3);
    grad_output << 5.0, 2.0, 3.0,
                   4.0, 1.0, 2.0;
    MatrixXd returned_grad(2, 3);
    {
        USING_TRAINING_MODE _;
        MatrixXd output = relu.forward(x);
        MatrixXd expected_output(2, 3);
        expected_output << 2.0, 0.0, 0.0,
                           1.5, 0.0, 0.0;
        ASSERT_TRUE(expected_output.isApprox(output));
        returned_grad = relu.backward(grad_output);
    }
    // ReLU derivative is 1 for positive inputs and 0 for non-positive inputs (negative or zero).
    // Thus, gradients should only pass through where the inputs were positive.
    // For [2.0, -3.0, 0.0], the gradient [5.0, 2.0, 3.0] should be gated to [5.0, 0.0, 0.0].
    // For [1.5, 0.0, -2.0], the gradient [4.0, 1.0, 2.0] should be gated to [4.0, 0.0, 0.0].
    MatrixXd expected_grad(2, 3);
    expected_grad << 5.0, 0.0, 0.0,
                     4.0, 0.0, 0.0;
    ASSERT_TRUE(expected_grad.isApprox(returned_grad));
}
