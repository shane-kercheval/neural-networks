#include <iostream>
#include "torchcpp.h"

int main() {
    // print hello world

    MatrixXd training_images;
    VectorXi training_labels;
    std::cout << "Loading training images/labels" << std::endl;
    torchcpp_data::load_mnist_data(
        training_images,
        training_labels,
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte"
    );
    std::cout << "Training images: " << training_images.rows() << std::endl;
    std::cout << "Training labels: " << training_labels.size() << std::endl;

    MatrixXd test_images;
    VectorXi test_labels;
    std::cout << "Loading test images/labels" << std::endl;
    torchcpp_data::load_mnist_data(
        test_images,
        test_labels,
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );
    std::cout << "Test images: " << test_images.rows() << std::endl;
    std::cout << "Test labels: " << test_labels.size() << std::endl;
}
