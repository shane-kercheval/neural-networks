#include "gtest/gtest.h"
#include "torchcpp.h"
#include "test_helpers.h"

using Eigen::MatrixXd;

TEST(ModuleTests, training_model_is_correctly_toggled) {
    // training model should be off
    ASSERT_FALSE(torchcpp::Module::training);
    {
        // training mode should be on within this block
        torchcpp::USING_TRAINING_MODE _;
        ASSERT_TRUE(torchcpp::Module::training);
    }
    // training model should be switched off automatically after the block
    ASSERT_FALSE(torchcpp::Module::training);
}

TEST(ModuleTests, foward_method_succeeds_when_training_mode_is_off) {
    tests::MockModule module;
    MatrixXd input(2, 2);
    input << 1, 2,
             3, 4;
    MatrixXd output = module.forward(input);
    ASSERT_TRUE(output.isApprox(input));
    MatrixXd output2 = module(input);
    ASSERT_TRUE(output2.isApprox(input));
}

TEST(ModuleTests, foward_method_succeeds_when_training_mode_is_on) {
    tests::MockModule module;
    MatrixXd input(2, 2);
    input << 1, 2,
             3, 4;
    MatrixXd output;
    {
        torchcpp::USING_TRAINING_MODE _;
        output = module.forward(input);
    }
    ASSERT_TRUE(output.isApprox(input));
    MatrixXd output2;
    {
        torchcpp::USING_TRAINING_MODE _;
        output2 = module(input);
    }
    ASSERT_TRUE(output2.isApprox(input));
}

TEST(ModuleTests, backwards_method_succeeds_when_training_mode_is_on) {
    tests::MockModule module;
    MatrixXd grad_output(2, 2);
    grad_output << 1, 2,
                   3, 4;
    MatrixXd grad_input;
    {
        torchcpp::USING_TRAINING_MODE _;
        grad_input = module.backward(grad_output);
    }
    ASSERT_TRUE(grad_input.isApprox(grad_output));
}

TEST(ModuleTests, backwards_method_fails_when_training_mode_is_off) {
    tests::MockModule module;
    MatrixXd grad_output(2, 2);
    grad_output << 1, 2,
                   3, 4;
    ASSERT_THROW(module.backward(grad_output), std::logic_error);
}

TEST(SequentialTest, test_linear_relu_sequential_forward) {
    const int num_samples = 4;
    const int num_features = 3;
    const int num_classes = 2;
    MatrixXd input(num_samples, num_features);
    input << 0.1, 1.2, 0.3,
             2.0, -1.0, 0.0,
             -0.5, 0.0, 1.5,
             0.3, 0.7, -1.5;

    std::vector<torchcpp::Module*> layers = {
        new torchcpp::Linear(num_features, 5),
        new torchcpp::ReLU(),
        new torchcpp::Linear(5, num_classes)
    };
    torchcpp::Sequential model(layers);
    MatrixXd output = model.forward(input);
    ASSERT_EQ(output.rows(), num_samples);
    ASSERT_EQ(output.cols(), num_classes);
    for (torchcpp::Module* module : layers) {
        delete module;  // NOLINT(cppcoreguidelines-owning-memory)
    }
}

TEST(SequentialTest, test_linear_relu_sequential_backwards) {
    const int num_samples = 4;
    const int num_features = 3;
    const int num_classes = 2;
    MatrixXd input(num_samples, num_features);
    input << 0.1, 1.2, 0.3,
             2.0, -1.0, 0.0,
             -0.5, 0.0, 1.5,
             0.3, 0.7, -1.5;

    std::vector<torchcpp::Module*> layers = {
        new tests::TestableLinear(num_features, 5),
        new torchcpp::ReLU(),
        new tests::TestableLinear(5, num_classes)
    };
    torchcpp::Sequential model(layers);

    MatrixXd original_weights = dynamic_cast<tests::TestableLinear*>(layers[0])->get_weights();
    MatrixXd original_biases = dynamic_cast<tests::TestableLinear*>(layers[0])->get_biases();
    MatrixXd original_weight_grads = dynamic_cast<tests::TestableLinear*>(layers[0])->get_weight_grad();
    MatrixXd original_bias_grads = dynamic_cast<tests::TestableLinear*>(layers[0])->get_bias_grad();

    // weight and bias grads should be zero before backpropagation
    ASSERT_TRUE(original_weight_grads.isZero());
    ASSERT_TRUE(original_bias_grads.isZero());

    {
        torchcpp::USING_TRAINING_MODE _;
        MatrixXd output = model.forward(input);
        ASSERT_EQ(output.rows(), num_samples);
        ASSERT_EQ(output.cols(), num_classes);

        MatrixXd grad_output = MatrixXd::Ones(num_samples, num_classes);
        MatrixXd grad_input = model.backward(grad_output);
        ASSERT_EQ(grad_input.rows(), num_samples);
        ASSERT_EQ(grad_input.cols(), num_features);

        // At this point, the gradients should be calculated and will be different from the original
        ASSERT_FALSE(dynamic_cast<tests::TestableLinear*>(layers[0])->get_weight_grad().isZero());
        ASSERT_FALSE(dynamic_cast<tests::TestableLinear*>(layers[0])->get_bias_grad().isZero());

        model.step(torchcpp::SGD(0.1));
        // after stepping, the weights should be updated
        ASSERT_FALSE(dynamic_cast<tests::TestableLinear*>(layers[0])->get_weights().isApprox(original_weights));
        ASSERT_FALSE(dynamic_cast<tests::TestableLinear*>(layers[0])->get_biases().isApprox(original_biases));
    }
    for (torchcpp::Module* module : layers) {
        delete module;  // NOLINT(cppcoreguidelines-owning-memory)
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
