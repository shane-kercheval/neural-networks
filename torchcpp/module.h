
#pragma once
#include <stdexcept>
#include <Eigen/Dense>

namespace torchcpp {

    /**
     * @brief Base class for all neural network components/modules.
     */
    class Module {
    public:
        inline static bool training = false;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

        Module() = default; // default constructor
        virtual ~Module() = default;  // default destructor

        /**
         * @brief Perform the forward pass of the module.
         *
         * @param x The input data.
         * @return The computed output.
        */
        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) = 0;  // = 0 means pure virtual function in C++ which means that the derived class must implement the function.
        
        /**
         * @brief Perform the backward pass of the module, calculating/returning the gradient with
         * respect to input.
         *   
         * Backpropagation is essentially an application of the chain rule, used to compute
         * gradient of a loss function with respect to the weights and biases of a neural network.
         * The gradient tells us how much (and in which direction) each parameter affects the loss
         * function (which we hope to minimize). Each layer's backward pass computes the gradients
         * (i.e. the partial derivatives) of the loss function relative to its inputs and
         * parameters, using the gradient that flows back from the subsequent layers.
         *
         * ∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W
         *
         * ∂L/∂W is the gradient (partial derivative) of the loss function (L) with respect to the
         * weights (W) in this layer. This is what tells us how to adjust the weights of this layer
         * to minimize the loss function.
         *
         * ∂L/∂a (i.e. grad_output) is the gradient of the loss function (L) with respect to the
         * output of the layer. This is calculated in the subsequent layer (or directly from the
         * loss function if it's the final layer) and passed back to this layer.
         *
         * ∂a/∂z is the gradient of the output with respect to the activation function, which is
         * often straightforward to compute (e.g., for ReLU, sigmoid).
         *
         * ∂z/∂W is the gradient of the layer's output with respect to its weights, which typically
         * involves the input to the layer, depending on whether it's fully connected,
         * convolutional, etc.
         *
         * @param grad_output Gradient of the loss with respect to the output of this module.
         * @return Gradient of the loss with respect to the input of this module.
        */
        Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) {
            if (!training) {
                throw std::logic_error("Module::backward should only be called during training");
            }
            return backward_impl(grad_output);
        };

        /**
         * @brief Update the parameters of the module using the gradient computed during
         * backpropagation and the optimizer provided.
         *
         * Only applicable to modules that require gradient computation (TrainableParamsModule)
         * but needs to be defined in the base class to avoid checking the type of module in the
         * training loop.
         *
         * @param optimizer The optimizer to use for updating the weights and biases. The optimizer
         * is a function that takes the parameters (of the Module) as the first function parameter
         * and the gradients as the second function parameter and updates the Module parameters.
        */
        void step(const std::function<void(Eigen::MatrixXd&, Eigen::MatrixXd&)>& optimizer) {
            if (!training) {
                throw std::logic_error("Module::step should only be called during training");
            }
            step_imp(optimizer);
            zero_grad();
        }

        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x) {
            // () operator overload; equivalent to `__call__` in python
            return forward(x);
        }

        // for now, we will disable copy and move operations for all modules; i'd like to
        // understand when they would be used before enabling them.
        // copy constructor
        Module(const Module& other) = delete;
        // copy assignment operator
        Module& operator=(const Module& other) = delete;
        // move constructor
        // by marking these functions as noexcept, we're indicating that they won't throw exceptions.
        // This is important for maintaining strong exception safety (i.e., the program remains in
        // a valid state and no resources are leaked if an exception is thrown).
        Module(Module&& other) noexcept = delete;
        // move assignment operator
        Module& operator=(Module&& other) noexcept = delete;

    protected:
        virtual Eigen::MatrixXd backward_impl(const Eigen::MatrixXd& grad_output) = 0;
        /**
         * @brief Update the parameters of the module using the gradient computed during
         * backpropagation and the optimizer provided.
         *
         * Note to self: step_imp is not a "pure" virtual function because it has a default
         * implementation {} (which doesn't do anything); not all modules will need to implement this
         * function (only those that have parameters that need to be updated). However, all modules
         * will inherit this function so that the training loop can call it without knowing whether
         * the module has parameters or not.
        */
        virtual void step_imp(const std::function<void(Eigen::MatrixXd&, Eigen::MatrixXd&)>& optimizer) {}
        /**
         * Zero the gradients of the parameters of the module. Models that require gradient
         * computation should override this method.
        */
        virtual void zero_grad() {}
    };

    /**
     * A context manager to set the module in training mode temporarily.
    */
    class USING_TRAINING_MODE {
    public:
        // constructor
        USING_TRAINING_MODE() {
            Module::training = true;
        }
        // destructor
        ~USING_TRAINING_MODE() {
            Module::training = false;
        }
    
        // for now, we will disable copy and move operations for all modules; i'd like to
        // understand when they would be used before enabling them.
        USING_TRAINING_MODE(const USING_TRAINING_MODE& other) = delete;  // copy Constructor
        USING_TRAINING_MODE& operator=(const USING_TRAINING_MODE& other) = delete;  // copy assignment operator
        USING_TRAINING_MODE(USING_TRAINING_MODE&& other) noexcept = delete;  // move constructor
        USING_TRAINING_MODE& operator=(USING_TRAINING_MODE&& other) noexcept = delete;  // move assignment operator
    };
}
