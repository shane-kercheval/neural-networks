#include "activations.h"

namespace torchcpp {

MatrixXd ReLU::forward(const MatrixXd& x) {
    if (Module::training) {
        output_cache_ = x.cwiseMax(0.0);;
        return output_cache_;
    }
    return x.cwiseMax(0.0);
}

MatrixXd ReLU::backward_impl(const MatrixXd& grad_output) {
    // Python code: return grad_output * (self.output > 0)  # Element-wise multiplication
    // The `.array()` method is used to switch the context in which Eigen treats the data. By
    // converting the matrix to an array, Eigen applies operations element-wise rather than
    // via matrix multiplication. The python code above also uses element-wise multiplication. 
    return grad_output.array() * (output_cache_.array() > 0).cast<double>();
}

}
