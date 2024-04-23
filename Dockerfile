FROM python:3.11

# Install necessary packages for building software
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    zsh \
    lsof \
    gcc \
    g++ \
    make \
    cmake \
    curl \
    libgtest-dev \
    libeigen3-dev \
    clang-tidy

# Build and install Google Test
RUN cd /usr/src/gtest && \
    cmake CMakeLists.txt && \
    make && \
    cp lib/*.a /usr/lib

# Set default shell to zsh (optional, depends on your preference)
RUN echo "dash dash/sh boolean false" | debconf-set-selections && dpkg-reconfigure dash

# Clean up to reduce image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code
ENV PYTHONPATH "${PYTHONPATH}:/code"

ENV PYDEVD_WARN_EVALUATION_TIMEOUT 1000

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

RUN pip3 install torch torchvision torchaudio

# Download MNIST dataset
RUN curl -o /code/data/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz && \
    curl -o /code/data/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && \
    curl -o /code/data/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz && \
    curl -o /code/data/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
# Unzip MNIST dataset
RUN apt-get install -y gzip && \
    gzip -d /code/data/*.gz
