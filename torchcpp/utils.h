#pragma once
#include <cmath>
#include <fstream>
#include <Eigen/Dense>


const int MAGIC_NUMBER_IMAGES = 2051;
const int MAGIC_NUMBER_LABELS = 2049;
const int MNIST_IMAGE_HEIGHT = 28;
const int MNIST_IMAGE_WIDTH = 28;


namespace torchcpp {

    /**
    He initialization is a way to initialize the weights of a neural network in a way that
    prevents the signal from vanishing or exploding as it passes through the network.

    The factor is calculated as sqrt(2. / in_features) for ReLU activations, and sqrt(1. /
    in_features) for other activations. We use the ReLU variant here.

    @param in_features
        The number of input features.
    @param out_features
        The number of output features. Unused in this initialization method. But is
        included for compatibility with other initialization methods.
    @return
        The scaling factor for the weights.
    */
    double he_init_scale(int in_features, int /*out_features*/) {
        return std::sqrt(2.0 / in_features);
    }

    /**
    Glorot initialization (also known as Xavier initialization) is a way to initialize the weights
    of a neural network in a way that prevents the signal from vanishing or exploding as it passes
    through the network.

    The factor is calculated as sqrt(6. / (in_features + out_features)).

    @param in_features
        The number of input features.
    @param out_features
        The number of output features.
    @return
        The scaling factor for the weights.
    */
    double glorot_init_scale(int in_features, int out_features) {
        return std::sqrt(6.0 / (in_features + out_features));
    }

}

namespace torchcpp_data {
    /**
    Load MNIST data from the specified path.

    Example usage:
    @code
    std::vector<Eigen::VectorXd> images;
    std::vector<int> labels;
    load_mnist_data(
        images,
        labels,
        "/code/data/train-images-idx3-ubyte",
        "/code/data/train-labels-idx1-ubyte",
        1000
    )
    @endcode

    @param images
        A vector of Eigen vectors where each vector will hold the pixels of an MNIST image.
    @param labels
        A single vector of integers where the MNIST labels will be stored correpsonding to each
        image in the images vector.
    @param image_path
        The path to the MNIST images.
    @param label_path
        The path to the MNIST labels.
    @param num_images
        The number of images to load. The default is -1 which means all images will be loaded.
    */
    void load_mnist_data(
            std::vector<Eigen::VectorXd>& images,
            std::vector<int>& labels,
            const std::string& image_path,
            const std::string& label_path,
            int num_images = -1
            ) {
        std::ifstream image_file(image_path, std::ios::binary);
        std::ifstream label_file(label_path, std::ios::binary);

        if (!image_file.is_open()) {
            throw std::runtime_error("Could not open image file: " + image_path);
        }
        if (!label_file.is_open()) {
            throw std::runtime_error("Could not open label file: " + label_path);
        }
        // Read the magic numbers for the images and labels
        // The magic numbers are the first 4 bytes of the file and are used to verify that we are
        // reading the correct file format.
        int magic_number_images = 0;
        int magic_number_labels = 0;
        image_file.read(reinterpret_cast<char*>(&magic_number_images), sizeof(magic_number_images));
        label_file.read(reinterpret_cast<char*>(&magic_number_labels), sizeof(magic_number_labels));
        // Convert the magic number from big endian to little endian.
        // This conversion is necessary because the MNIST data format uses big endian, whereas most
        // computers use little endian. The __builtin_bswap32 is a GCC built-in function to swap
        // byte order.
        magic_number_images = __builtin_bswap32(magic_number_images);
        magic_number_labels = __builtin_bswap32(magic_number_labels);

        if (magic_number_images != MAGIC_NUMBER_IMAGES) {
            throw std::runtime_error("Invalid MNIST image file!");
        }
        if (magic_number_labels != MAGIC_NUMBER_LABELS) {
            throw std::runtime_error("Invalid MNIST label file!");
        }

        // read the number of images and labels, which are 4 bytes each and are also in big endian
        // format.
        int num_images_in_file = 0;
        int num_labels_in_file = 0;
        image_file.read(reinterpret_cast<char*>(&num_images_in_file), sizeof(num_images_in_file));
        label_file.read(reinterpret_cast<char*>(&num_labels_in_file), sizeof(num_labels_in_file));
        num_images_in_file = __builtin_bswap32(num_images_in_file);
        num_labels_in_file = __builtin_bswap32(num_labels_in_file);

        // ensure the number of images and labels are the same and not equal to 0
        if (num_images_in_file != num_labels_in_file || num_images_in_file == 0) {
            throw std::runtime_error("Invalid MNIST data files!");
        }
        if (num_images == -1 || num_images > num_images_in_file) {
            num_images = num_images_in_file;
        }
        // preallocate the memory for the images and labels rather than resizing the vectors each
        // time we read an image in the for loop below.
        images.reserve(num_images);
        labels.reserve(num_images);

        // read the images and labels
        // define an Eigen vector to hold the pixels of an image
        Eigen::VectorXd image(MNIST_IMAGE_HEIGHT * MNIST_IMAGE_WIDTH);
        for (int i = 0; i < num_images; ++i) {
            unsigned char temp = 0;

            label_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
            labels.push_back(static_cast<int>(temp));
            // read the pixels of the image
            for (int p = 0; p < MNIST_IMAGE_HEIGHT * MNIST_IMAGE_WIDTH; ++p) {
                image_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
                // normalize the pixel values to be between 0 and 1
                image(p) = static_cast<double>(temp) / 255.0;
            }
            images.push_back(image);
        }
    }
}
