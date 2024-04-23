#pragma once
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace torchcpp {

    double he_init_scale(int in_features, int out_features);
    double glorot_init_scale(int in_features, int out_features);

}

namespace torchcpp_data {
    void load_mnist_data(
        std::vector<Eigen::VectorXd>& images,
        std::vector<int>& labels,
        const std::string& image_path,
        const std::string& label_path,
        unsigned int num_images = 0
    );
}
