#include "gtest/gtest.h"
#include "torchcpp/torchcpp.h"


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
TEST(UtilsTest, load_mnist_data) {
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
    EXPECT_EQ(labels[0], 5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
