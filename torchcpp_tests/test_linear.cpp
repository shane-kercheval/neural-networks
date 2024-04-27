#include <gtest/gtest.h>
#include <functional>
#include <Eigen/Dense>
#include "torchcpp.h"
#include "test_helpers.h"

using namespace torchcpp;
using Eigen::MatrixXd;


/**
 * This test checks that the forward pass of the Linear layer is correct. It does this by
 * initializing the weights/biases to 0, and then shifting them by 1 (via step()).
 * It then passes an input matrix to the forward method, and checks that the output is as expected
 * based on multiplying the input by the weights and adding the biases. This also tests the step()
 * method.
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
    // ensure .forward() works as expected
    ASSERT_TRUE(expected_output.isApprox(linear.forward(x)));
    // ensure calling the object as a function works the same way
    ASSERT_TRUE(expected_output.isApprox(linear(x)));
}

/**
 * This test checks that the backward pass of the Linear layer returns the correct shape of the
 * gradient.
*/
TEST(TestLinear, backward_pass_correct_shape) {
    Linear linear(3, 2);  // 3 input features, 2 output features
    MatrixXd x(4, 3);  // 4 samples, 3 features
    x << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12;
    MatrixXd grad_output(4, 2);  // 4 samples, 2 output features
    grad_output << 1, 2,
                   3, 4,
                   5, 6,
                   7, 8;
    MatrixXd grad_input;
    {
        USING_TRAINING_MODE _;
        MatrixXd output = linear.forward(x);
        grad_input = linear.backward(grad_output);
        // ensure step works
        linear.step([](MatrixXd& weights, MatrixXd& grad) {
            weights += grad;
        });
    }
    ASSERT_EQ(grad_input.rows(), 4);
    ASSERT_EQ(grad_input.cols(), 3);
}

/**
 * This tests performs numerical gradient checking on a Linear layer via numerical approximation.
 * This is a technique to approximate the gradient computed by backpropagation and compare it with
 * the gradient calculated using a numerical approximation method. This test ensures the
 * backpropagation implementation is correct.
*/
TEST(TestLinear, backward_pass_using_numerical_approximation) {
    const int batch_size = 4;
    const int input_features = 3;
    const int output_features = 2;
    const double epsilon = 1e-5;
    const int expected_iterations = input_features * output_features;
    tests::TestableLinear linear(input_features, output_features);

    ASSERT_EQ(linear.get_weights().rows(), input_features);
    ASSERT_EQ(linear.get_weights().cols(), output_features);

    MatrixXd x = Eigen::MatrixXd::Random(batch_size, input_features);
    MatrixXd fake_grad_ouput = Eigen::MatrixXd::Ones(batch_size, output_features);

    int actual_iterations = 0;
    {
        USING_TRAINING_MODE _;
        for (int i = 0; i < input_features; ++i) {
            for (int j = 0; j < output_features; ++j) {
                double original_weight = linear.get_weights()(i, j);
                // increase the current weight by a small value and compute the output
                linear.get_weights()(i, j) = original_weight + epsilon;
                MatrixXd output_plus = linear.forward(x);
                // decrease the current weight by a small value and compute the output
                linear.get_weights()(i, j) = original_weight - epsilon;
                MatrixXd output_minus = linear.forward(x);
                // compute the numerical gradient
                double estimated_gradient = (output_plus - output_minus).sum() / (2 * epsilon);
                // reset the weight to its original value
                linear.get_weights()(i, j) = original_weight;
                // compute the gradient using backpropagation
                linear.backward(fake_grad_ouput);
                double actual_gradient = linear.get_weight_grad()(i, j);
                // verify that the computed gradient matches the numerical estimate
                ASSERT_NEAR(estimated_gradient, actual_gradient, epsilon);
                actual_iterations++;
            }
        }

    }
    // ensure we are testing the correct number of iterations (i.e. all weights)
    ASSERT_EQ(expected_iterations, actual_iterations);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
