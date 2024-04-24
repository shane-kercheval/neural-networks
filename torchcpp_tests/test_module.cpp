#include "gtest/gtest.h"
#include "torchcpp.h"
#include "test_helpers.h"


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
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
             3, 4;
    Eigen::MatrixXd output = module.forward(input);
    ASSERT_TRUE(output.isApprox(input));
    Eigen::MatrixXd output2 = module(input);
    ASSERT_TRUE(output2.isApprox(input));
}

TEST(ModuleTests, foward_method_succeeds_when_training_mode_is_on) {
    tests::MockModule module;
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
             3, 4;
    Eigen::MatrixXd output;
    {
        torchcpp::USING_TRAINING_MODE _;
        output = module.forward(input);
    }
    ASSERT_TRUE(output.isApprox(input));
    Eigen::MatrixXd output2;
    {
        torchcpp::USING_TRAINING_MODE _;
        output2 = module(input);
    }
    ASSERT_TRUE(output2.isApprox(input));
}

TEST(ModuleTests, backwards_method_succeeds_when_training_mode_is_on) {
    tests::MockModule module;
    Eigen::MatrixXd grad_output(2, 2);
    grad_output << 1, 2,
                   3, 4;
    Eigen::MatrixXd grad_input;
    {
        torchcpp::USING_TRAINING_MODE _;
        grad_input = module.backward(grad_output);
    }
    ASSERT_TRUE(grad_input.isApprox(grad_output));
}

TEST(ModuleTests, backwards_method_fails_when_training_mode_is_off) {
    tests::MockModule module;
    Eigen::MatrixXd grad_output(2, 2);
    grad_output << 1, 2,
                   3, 4;
    ASSERT_THROW(module.backward(grad_output), std::logic_error);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
