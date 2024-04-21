#include "gtest/gtest.h"
#include "torchcpp/torchcpp.h"


// Testing He initialization scale
TEST(UtilsTest, HeInitScale) {
    int ignored = 0;
    EXPECT_NEAR(torchcpp::he_init_scale(10, ignored), std::sqrt(2.0 / 10), 0.001);
}

// Testing Glorot initialization scale
TEST(UtilsTest, GlorotInitScale) {
    EXPECT_NEAR(torchcpp::glorot_init_scale(8, 12), std::sqrt(6.0 / (8 + 12)), 0.001);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
