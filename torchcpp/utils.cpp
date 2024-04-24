#include "utils.h"
#include <stdexcept>


const int MAGIC_NUMBER_IMAGES = 2051;
const int MAGIC_NUMBER_LABELS = 2049;
const int MNIST_IMAGE_HEIGHT = 28;
const int MNIST_IMAGE_WIDTH = 28;

const double HE_FACTOR = 2.0;
const double GLOROT_FACTOR = 6.0;
const double MNIST_PIXEL_MAX = 255.0;


namespace torchcpp {

    double he_init_scale(int in_features, int /*out_features*/) {
        return std::sqrt(HE_FACTOR / in_features);
    }

    double glorot_init_scale(int in_features, int out_features) {
        return std::sqrt(GLOROT_FACTOR / (in_features + out_features));
    }

}

namespace torchcpp_data {

    void load_mnist_data(
            std::vector<Eigen::VectorXd>& images,
            std::vector<int>& labels,
            const std::string& image_path,
            const std::string& label_path,
            unsigned int num_images
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
        unsigned int magic_number_images = 0;
        unsigned int magic_number_labels = 0;
        image_file.read(reinterpret_cast<char*>(&magic_number_images), sizeof(magic_number_images));  // NOLINT
        label_file.read(reinterpret_cast<char*>(&magic_number_labels), sizeof(magic_number_labels));  // NOLINT
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
        unsigned int num_images_in_file = 0;
        unsigned int num_labels_in_file = 0;
        image_file.read(reinterpret_cast<char*>(&num_images_in_file), sizeof(num_images_in_file));  // NOLINT
        label_file.read(reinterpret_cast<char*>(&num_labels_in_file), sizeof(num_labels_in_file));  // NOLINT
        num_images_in_file = __builtin_bswap32(num_images_in_file);
        num_labels_in_file = __builtin_bswap32(num_labels_in_file);

        // ensure the number of images and labels are the same and not equal to 0
        if (num_images_in_file != num_labels_in_file || num_images_in_file == 0) {
            throw std::runtime_error("Invalid MNIST data files!");
        }
        if (num_images == 0 || num_images > num_images_in_file) {
            num_images = num_images_in_file;
        }
        // preallocate the memory for the images and labels rather than resizing the vectors each
        // time we read an image in the for loop below.
        images.reserve(num_images);
        labels.reserve(num_images);

        // read the images and labels
        // define an Eigen vector to hold the pixels of an image
        Eigen::VectorXd image(MNIST_IMAGE_HEIGHT * MNIST_IMAGE_WIDTH);
        for (unsigned int i = 0; i < num_images; ++i) {
            unsigned char temp = 0;

            label_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));  // NOLINT
            labels.push_back(static_cast<int>(temp));
            // read the pixels of the image
            for (int p = 0; p < MNIST_IMAGE_HEIGHT * MNIST_IMAGE_WIDTH; ++p) {
                image_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));  // NOLINT
                // normalize the pixel values to be between 0 and 1
                image(p) = static_cast<double>(temp) / MNIST_PIXEL_MAX;
            }
            images.push_back(image);
        }
    }
}
