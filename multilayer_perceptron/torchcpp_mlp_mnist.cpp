#include <iostream>
#include <chrono>
#include "torchcpp.h"

const double LEARNING_RATE = 0.01;
const int NUM_EPOCHS = 15;
const long BATCH_SIZE = 32;  // long because .rows() returns a long and we don't want to cast each time
const int NUM_FEATURES = 28 * 28;
const int NUM_CLASSES = 10;


using Eigen::MatrixXd;
using Eigen::VectorXi;

int main() {
    // read in test dataset
    MatrixXd test_images;
    VectorXi test_labels;
    std::cout << "Loading test images/labels" << std::endl;
    torchcpp_data::load_mnist_data(
        test_images,
        test_labels,
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );
    std::cout << "test_images: " << test_images.rows() << " x " << test_images.cols() << std::endl; 
    std::cout << "test_labels: " << test_labels.size() << std::endl;

    // read in training dataset; we need to split this into training and validation sets
    MatrixXd training_images;
    VectorXi training_labels;
    MatrixXd validation_images;
    VectorXi validation_labels;
    {
        // new scope to avoid polluting the global namespace and to free up memory after copying
        // training and validation sets
        MatrixXd init_training_images;
        VectorXi init_training_labels;
        std::cout << "Loading training images/labels" << std::endl;
        torchcpp_data::load_mnist_data(
            init_training_images,
            init_training_labels,
            "data/train-images-idx3-ubyte",
            "data/train-labels-idx1-ubyte"
        );
        std::cout << "Initial Training images: " << init_training_images.rows() << std::endl;
        std::cout << "Initial Training labels: " << init_training_labels.size() << std::endl;
        // create validation set by taking the first 7,000 samples from the training set
        validation_images = init_training_images.topRows(7000);
        validation_labels = init_training_labels.head(7000);
        std::cout << "validation_images: " << validation_images.rows() << " x " << validation_images.cols() << std::endl;
        std::cout << "validation_labels: " << validation_labels.size() << std::endl;
        training_images = init_training_images.bottomRows(init_training_images.rows() - 7000);
        training_labels = init_training_labels.tail(init_training_labels.size() - 7000);
        std::cout << "training_images: " << training_images.rows() << " x " << training_images.cols() << std::endl;
        std::cout << "training_labels: " << training_labels.size() << std::endl;
    }
    
    // TODO: shuffle datasets; perhaps do it in load_mnist_data function

    torchcpp::CrossEntropyLoss loss_function;
    torchcpp::SGD optimizer(LEARNING_RATE);

    std::vector<torchcpp::Module*> layers = {
        new torchcpp::Linear(NUM_FEATURES, 100, torchcpp::he_init_scale),
        new torchcpp::ReLU(),
        new torchcpp::Linear(100, NUM_CLASSES, torchcpp::glorot_init_scale)
    };
    torchcpp::Sequential model(layers);

    // training loop
    double loss = 0.0;
    double validation_loss = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();
    int actual_batch_size = 0;
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        for (int batch_index = 0; batch_index < training_images.rows(); batch_index += BATCH_SIZE) {
            // std::cout << "Epoch: " << epoch << ", Batch: " << batch_index << std::endl;
            // we need to handle the case where the last batch is smaller than BATCH_SIZE
            actual_batch_size = std::min(BATCH_SIZE, training_images.rows() - batch_index);
            MatrixXd x_batch = training_images.middleRows(batch_index, actual_batch_size);
            VectorXi y_batch = training_labels.segment(batch_index, actual_batch_size);
            {
                torchcpp::USING_TRAINING_MODE _;  // start training mode

                MatrixXd logits = model.forward(x_batch);
                loss = loss_function.forward(logits, y_batch);
                MatrixXd loss_grad = loss_function.backward();
                model.backward(loss_grad);
                model.step(optimizer);
            }
            if (batch_index / BATCH_SIZE % 400 == 0) {
                MatrixXd validation_logits = model.forward(validation_images);
                validation_loss = loss_function.forward(validation_logits, validation_labels);
                std::cout << "Epoch " << epoch << ", Batch " << batch_index / BATCH_SIZE << 
                    ", training loss " << loss << ", validation loss " << 
                    validation_loss << std::endl;
            }

        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    // Test loss on test set
    MatrixXd test_logits = model.forward(test_images);
    double test_loss = loss_function.forward(test_logits, test_labels);
    std::cout << "Test loss: " << test_loss << std::endl;

    // Test accuracy on test set
    MatrixXd probabilities = torchcpp::softmax(test_logits);
    // get the index of the highest probability in each row 
    Eigen::VectorXi predicted_indices = Eigen::VectorXi::Zero(test_logits.rows());
    for (int i = 0; i < test_logits.rows(); ++i) {
        Eigen::VectorXd::Index max_index;
        test_logits.row(i).maxCoeff(&max_index);
        predicted_indices(i) = max_index;
    }
    double accuracy = (predicted_indices.array() == test_labels.array()).cast<double>().mean();
    std::cout << "Test accuracy: " << accuracy << std::endl;

    return 0;
}
