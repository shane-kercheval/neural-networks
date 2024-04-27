#include "gtest/gtest.h"
#include "torchcpp.h"
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

TEST(SoftmaxTest, softmax) {
    MatrixXd logits(3, 3);
    VectorXi targets(3);
    logits << 0.1, 1.2, 0.3,
              2.0, -1.0, 0.0,
              -0.5, 0.0, 1.5;
    targets << 1, 0, 2;
    // expcted probabilities calculated from python unit test
    // array([[0.1913667280438832, 0.5748974225032308, 0.23373584945288603],
    //        [0.8437947344813396, 0.04201006613406606, 0.1141951993845945],
    //        [0.09962364806231833, 0.16425162762508783, 0.7361247243125939]])
    MatrixXd expected_probabilities(3, 3);
    expected_probabilities << 0.1913667280438832, 0.5748974225032308, 0.23373584945288603,
                              0.8437947344813396, 0.04201006613406606, 0.1141951993845945,
                              0.09962364806231833, 0.16425162762508783, 0.7361247243125939;
    ASSERT_TRUE(expected_probabilities.isApprox(torchcpp::softmax(logits)));
}

TEST(load_mnist_data, data_loads_successfully) {
    Eigen::MatrixXd images;
    VectorXi labels;
    torchcpp_data::load_mnist_data(
        images,
        labels,
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
        1000
    );
    EXPECT_EQ(images.rows(), 1000);
    EXPECT_EQ(images.cols(), 28 * 28);
    EXPECT_EQ(labels.size(), 1000);
    EXPECT_EQ(labels[0], 5);  // First label in the MNIST training dataset
    EXPECT_EQ(labels[1], 0);  // Second label in the MNIST training dataset
}

TEST(load_mnist_data, data_loads_successfully_all_images) {
    Eigen::MatrixXd images;
    VectorXi labels;
    // not specifying the number of images to load should load all images
    torchcpp_data::load_mnist_data(
        images,
        labels,
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );
    std::cout << images.rows() << std::endl;
    EXPECT_EQ(images.rows(), 10000);
    EXPECT_EQ(images.cols(), 28 * 28);
    EXPECT_EQ(labels.size(), 10000);
    EXPECT_EQ(labels[0], 7);  // First label in the MNIST test dataset
    EXPECT_EQ(labels[1], 2);  // Second label in the MNIST test dataset
}

TEST(load_mnist_data, fails_on_invalid_path) {
    Eigen::MatrixXd images;
    VectorXi labels;
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
