#include "torchcpp/torchcpp.h"

int main() {
    // Create a network using torchcpp components
    torchcpp::Linear layer(128, 64);
    double scale = torchcpp::he_init_scale(128);
    // More network setup and operations
}
