#pragma once
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXi;

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

    /**
     * Applies the softmax function to the logits and returns the probabilities
     * 
     * @param logits The logits (raw output from neural net) for each class. Each row is a sample
     * and each column is a class.
     * @return The probabilities for each class for each sample. The sum of each row will be 1.
    */
    MatrixXd softmax(const MatrixXd& logits);

}

namespace torchcpp_data {

    
    /**
     * Load MNIST data from the specified path.
     *
     * Example usage:
     * @code
     * std::vector<VectorXd> images;
     * VectorXi labels;
     * load_mnist_data(
     *    images,
     *    labels,
     *    "/code/data/train-images-idx3-ubyte",
     *    "/code/data/train-labels-idx1-ubyte",
     *    1000
     * )
     * @endcode
     *
     * @param images A matrx where each row will hold the pixels of an MNIST image and each column
     * will be a pixel value. So there will be 28 * 28 = 784 columns and the number of rows will be
     * equal to the number of images loaded.
     * @param labels A single vector of integers where the MNIST labels will be stored
     * correpsonding to each image in the images vector.
     * @param image_path The path to the MNIST images.
     * @param label_path The path to the MNIST labels.
     * @param num_images The number of images to load. The default is 0 which means all images
     * will be loaded.
     */
    void load_mnist_data(
        MatrixXd& images,
        VectorXi& labels,
        const std::string& image_path,
        const std::string& label_path,
        unsigned int num_images = 0
    );

}
