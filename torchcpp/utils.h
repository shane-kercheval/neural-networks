#pragma once
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace torchcpp {

    /**
     * He initialization is a way to initialize the weights of a neural network in a way that prevents the signal from vanishing or exploding as it passes through the network.
     * 
     * The factor is calculated as sqrt(2. / in_features) for ReLU activations, and sqrt(1. / in_features) for other activations. We use the ReLU variant here.
     * 
     * @param in_features The number of input features.
     * @param out_features The number of output features. Unused in this initialization method. But is included for compatibility with other initialization methods.
     * @return The scaling factor for the weights.
     */
    double he_init_scale(int in_features, int out_features);

    /**
     * Glorot initialization (also known as Xavier initialization) is a way to initialize the weights of a neural network in a way that prevents the signal from vanishing or exploding as it passes through the network.
     *
     * The factor is calculated as sqrt(6. / (in_features + out_features)).
     *
     * @param in_features The number of input features.
     * @param out_features The number of output features.
     * @return The scaling factor for the weights.
     */
    double glorot_init_scale(int in_features, int out_features);

}

namespace torchcpp_data {

    
    /**
     * Load MNIST data from the specified path.
     *
     * Example usage:
     * @code
     * std::vector<Eigen::VectorXd> images;
     * std::vector<int> labels;
     * load_mnist_data(
     *    images,
     *    labels,
     *    "/code/data/train-images-idx3-ubyte",
     *    "/code/data/train-labels-idx1-ubyte",
     *    1000
     * )
     * @endcode
     *
     * @param images A vector of Eigen vectors where each vector will hold the pixels of an MNIST image.
     * @param labels A single vector of integers where the MNIST labels will be stored correpsonding to each image in the images vector.
     * @param image_path The path to the MNIST images.
     * @param label_path The path to the MNIST labels.
     * @param num_images The number of images to load. The default is 0 which means all images will be loaded.
     */
    void load_mnist_data(
        std::vector<Eigen::VectorXd>& images,
        std::vector<int>& labels,
        const std::string& image_path,
        const std::string& label_path,
        unsigned int num_images = 0
    );

}
