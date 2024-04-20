# to run file:
#     > pip install line_profiler
#     > kernprof -l -v cnn/profile_cnn.py
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from line_profiler import profile


from torchpy import net

@profile
def main():
    SEED = 42
    DIMENSIONS = (28, 28)
    CHANNELS = 1  # grayscale
    INPUT_SIZE = DIMENSIONS[0] * DIMENSIONS[1]
    OUTPUT_SIZE = 10

    def transform_data(
            is_convolutional: bool,
            x: pd.DataFrame,
            y: pd.Series | None = None) -> tuple[np.array, np.array]:
        """
        Transforms the data. Returns a tuple of (x, y) where x is a tensor of the features and y is a
        tensor of the labels.

        Args:
            is_convolutional: A boolean indicating whether the model is convolutional.
            x: A dataframe of the features.
            y: A series of the labels.
        """
        x = x.to_numpy()
        # Normalize the tensor
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)
        assert x.min() == 0
        assert x.max() == 1
        if is_convolutional:
            # Reshape data to have channel dimension
            # MNIST images are 28x28, so we reshape them to [batch_size, 1, 28, 28]
            x = x.reshape(-1, CHANNELS, DIMENSIONS[0], DIMENSIONS[1])
        if y is not None:
            y = y.to_numpy().astype(int)
        return x, y


    def get_data():  # noqa
        """Function is required by and called from `model_pipeline()`."""
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
        x, y = transform_data(is_convolutional=True, x=x, y=y)
        # 80% train; 10% validation; 10% test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=SEED)  # noqa
        print(f"Training set  : X-{x_train.shape}, y-{y_train.shape}")
        print(f"Validation set: X-{x_val.shape}, y-{y_val.shape}")
        print(f"Test set      : X-{x_test.shape}, y-{y_test.shape}")
        return x_train, x_val, x_test, y_train, y_val, y_test

    x_train, x_val, x_test, y_train, y_val, y_test = get_data()


    learning_rate = 0.01
    batch_size = 32


    rng = np.random.default_rng(SEED)
    shuffle_index = rng.permutation(x_train.shape[0])
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

    import torchpy.architectures as arch
    model = arch.CNN()

    # Loss function and optimizer
    loss_function = net.CrossEntropyLoss()
    optimizer = net.SGD(learning_rate=learning_rate)

    training_losses = []
    validation_losses = []

    epoch = 0
    # batch = 0
    # i = 0
    for batch, i in enumerate(range(0, batch_size*2, batch_size)):
        # for epoch in range(1):
        # for batch, i in enumerate(range(0, batch_size*10, batch_size)):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        with net.training_mode():
            # Forward pass
            print("forward")
            logits = model(x_batch)
            print("loss")
            loss = loss_function(logits, y_batch)
            # Backward pass
            print("backward")
            loss_grad = loss_function.backward()
            model.backward(loss_grad)
            print("step")
            model.step(optimizer)
            print('done')
        training_losses.append(loss)
        # calculate validation loss
        # print("val_loss")
        # logits = model(x_val)
        # val_loss = loss_function(logits=logits, targets=y_val)
        # print("done val loss")
        # validation_losses.append(val_loss)
        # print(f"Epoch {epoch}, Batch {batch:04}, training loss {round(loss, 3)}, validation loss {round(val_loss, 3)}")  # noqa




if __name__ == '__main__':
    main()


#   import pandas as pd
# Training set  : X-(56000, 1, 28, 28), y-(56000,)
# Validation set: X-(7000, 1, 28, 28), y-(7000,)
# Test set      : X-(7000, 1, 28, 28), y-(7000,)
# forward
# loss
# backward
# step
# done
# val_loss
# done val loss
# Epoch 0, Batch 0000, training loss 2.322, validation loss 2.292
# Wrote profile results to profile.py.lprof
# Timer unit: 1e-06 s

# Total time: 28.9272 s
# File: /code/torchpy/net.py
# Function: forward at line 708

# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#    708                                               @profile
#    709                                               def forward(self, x):
#    710                                                   """
#    711                                                   Perform the forward pass of the convolutional layer using the input data.
#    712                                                   
#    713                                                   Args:
#    714                                                       x (ndarray): Input data of shape (batch_size, in_channels, height, width).
#    715                                                   
#    716                                                   Returns:
#    717                                                       ndarray: Output data after applying the convolution operation.
#    718                                                   """
#    719         4      54934.3  13733.6      0.2          self.x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')  # Pad the input
#    720         4          2.6      0.7      0.0          batch_size, _, H, W = self.x.shape  # Dimensions of the padded input
#    721         4          3.5      0.9      0.0          H_out = (H - self.kernel_size) // self.stride + 1  # Calculate output height
#    722         4          0.7      0.2      0.0          W_out = (W - self.kernel_size) // self.stride + 1  # Calculate output width
#    723         4         86.8     21.7      0.0          y = np.zeros((batch_size, self.out_channels, H_out, W_out))  # Initialize output tensor
#    724                                           
#    725        88          9.2      0.1      0.0          for i in range(H_out):
#    726      2044        257.9      0.1      0.0              for j in range(W_out):
#    727      1960        718.5      0.4      0.0                  h_start = i * self.stride  # Start index for height slicing
#    728      1960        401.0      0.2      0.0                  h_end = h_start + self.kernel_size  # End index for height slicing
#    729      1960        219.6      0.1      0.0                  w_start = j * self.stride  # Start index for width slicing
#    730      1960        273.4      0.1      0.0                  w_end = w_start + self.kernel_size  # End index for width slicing
#    731      1960       1292.0      0.7      0.0                  x_slice = self.x[:, :, h_start:h_end, w_start:w_end]  # Extract the relevant slice
#    732     39592       6849.8      0.2      0.0                  for k in range(self.out_channels):  # Iterate over each filter
#    733     37632   28862129.5    767.0     99.8                      y[:, k, i, j] = np.sum(x_slice * self.weights[k, :, :, :], axis=(1, 2, 3)) + self.biases[k]  # Convolve and add bias
#    734                                           
#    735         4         10.2      2.5      0.0          if Module.training:
#    736         2          1.2      0.6      0.0              self.output = y  # Store output for backpropagation
#    737         4          0.3      0.1      0.0          return y

# root@83c0f9aa6729:/code# 
