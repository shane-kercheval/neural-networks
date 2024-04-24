#include <Eigen/Dense>
#include <torchcpp/torchcpp.h>

namespace tests {

class MockModule : public torchcpp::Module {
public:
    MockModule() = default;

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) override {return x; }
    Eigen::MatrixXd backward_impl(const Eigen::MatrixXd& grad_output) override { return grad_output; }

    Eigen::MatrixXd weights;
    Eigen::MatrixXd weight_grad;
};

}
