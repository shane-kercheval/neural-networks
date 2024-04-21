#pragma once
#include <cmath>

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
