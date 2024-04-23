#include "gtest/gtest.h"
#include "torchcpp/torchcpp.h"
#include <Eigen/Dense>

// Testing He initialization scale
TEST(UtilsTest, he_init_scale) {
    int ignored = 0;
    EXPECT_NEAR(torchcpp::he_init_scale(10, ignored), std::sqrt(2.0 / 10), 0.001);
}

// Testing Glorot initialization scale
TEST(UtilsTest, glorot_init_scale) {
    EXPECT_NEAR(torchcpp::glorot_init_scale(8, 12), std::sqrt(6.0 / (8 + 12)), 0.001);
}

// Testing MNIST data loading
TEST(load_mnist_data, data_loads_successfully) {
    std::vector<Eigen::VectorXd> images;
    std::vector<int> labels;
    torchcpp_data::load_mnist_data(
        images,
        labels,
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
        1000
    );
    EXPECT_EQ(images.size(), 1000);
    EXPECT_EQ(labels.size(), 1000);
    EXPECT_EQ(images[0].size(), 28 * 28);
    EXPECT_EQ(labels[0], 5);  // First label in the MNIST dataset
    EXPECT_EQ(labels[1], 0);  // Second label in the MNIST dataset
}

TEST(load_mnist_data, fails_on_invalid_path) {
    std::vector<Eigen::VectorXd> images;
    std::vector<int> labels;
    EXPECT_THROW(
        torchcpp_data::load_mnist_data(
            images,
            labels,
            "data/invalid-path",
            "data/invalid-path",
            1000
        ),
        std::runtime_error
    );
}   

TEST(EigenOperationExample, ElementWiseOperation_add) {
    Eigen::MatrixXd matrix_a(2, 2);
    matrix_a << 1, 2,
                3, 4;
    Eigen::MatrixXd matrix_b(2, 2);
    matrix_b << 5, 6,
                7, 8;
    Eigen::MatrixXd expected(2, 2);
    expected << 6, 8,
                10, 12;
    Eigen::MatrixXd actual = matrix_a.array() + matrix_b.array();
    EXPECT_EQ(expected, actual);
}

TEST(EigenOperationExample, ElementWiseOperation_add_row_wise) {
    Eigen::MatrixXd matrix_a(2, 2);
    matrix_a << 1, 2,
                3, 4;
    Eigen::VectorXd expected(2);
    expected << 4, 6;
    Eigen::VectorXd actual = matrix_a.rowwise().sum();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
