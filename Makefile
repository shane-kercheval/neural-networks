####
# DOCKER
####
docker_build:
	docker compose -f docker-compose.yml build

docker_run: docker_build
	docker compose -f docker-compose.yml up

docker_down:
	docker compose down --remove-orphans

docker_rebuild:
	docker compose -f docker-compose.yml build --no-cache

####
# torchpy - Pytorch-like library
####
torchpy_linting:
	ruff check torchpy
	ruff check torchpy_tests

torchpy_unit_tests:
	pytest torchpy_tests

torchpy_tests: torchpy_linting torchpy_unit_tests

####
# torchcpp - Pytorch-like library for C++
####
.PHONY: all clean test run_mlp setup_build_dir compile torchcpp_tests torchcpp_unit_tests

CPP_SOURCE = ./torchcpp
CPP_TESTS = ./torchcpp_tests
SOURCE_FILES := $(shell find $(CPP_SOURCE) -name '*.cpp' -o -name '*.h')
TEST_FILES := $(shell find $(CPP_TESTS) -name '*.cpp' -o -name '*.h')

# Directory where the build files are located
BUILD_DIR := build

# Default target
all: compile

# Create the build directory and run cmake
setup_build_dir:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ..

# Compile the project using the generated Makefile by CMake
compile: setup_build_dir
	$(MAKE) -C $(BUILD_DIR)

# run linting
torchpp_lint:
	clang-tidy $(SOURCE_FILES) $(TEST_FILES) -p build/
# torchpp_lint:
# 	for file in $(SOURCE_FILES) $(TEST_FILES) ; do \
# 		clang-tidy $$file -- -std=c++20 -I/usr/include/eigen3 -I./ ; \
# 	done

# Run tests using CTest
torchcpp_unit_tests: compile
	cd $(BUILD_DIR) && ctest

torchcpp_tests: all torchpp_lint torchcpp_unit_tests

# Run the Multilayer Perceptron (MLP) executable
torchcpp_run_mlp: build
	./$(BUILD_DIR)/torchcpp_mlp

# Clean up the build directory
clean:
	rm -rf $(BUILD_DIR)

download_mnist:
	# Download MNIST dataset
	curl -o /code/data/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz && \
    curl -o /code/data/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && \
    curl -o /code/data/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz && \
    curl -o /code/data/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	# Unzip MNIST dataset
	apt-get install -y gzip && \
    	gzip -d /code/data/*.gz

####
# original/archived makefile in favor of CMake
###
# CPP_SOURCE = ./torchcpp
# CPP_TESTS = ./torchcpp_tests
# CPP_SOURCE_FILES := $(shell find $(CPP_SOURCE) -name '*.cpp')
# CPP_TEST_FILES := $(shell find $(CPP_TESTS) -name '*.cpp')
# CXX = g++

# # -Wall is to enable most warning messages from the compiler
# # -I is to add the incl ude directory to the compiler's search path; we add the current directory and the Google Test include directory
# # -pthread is to enable POSIX threads which is required by gtest
# CXXFLAGS = -std=c++20 -Wall -I./ -I/usr/include/gtest/ -pthread -I/usr/include/eigen3

# # GTEST_LIB contains the linker flags to link against the Google Test libraries.
# # Here it links against gtest_main and gtest. The gtest_main library provides a main function that
# # runs all tests, so I don't need to define it.
# GTEST_LIB = -lgtest_main -lgtest

# test_module: $(CPP_TESTS)/test_module.cpp $(CPP_SOURCE)/module.h $(CPP_SOURCE)/module.cpp
# 	$(CXX) $(CXXFLAGS) $(CPP_SOURCE)/module.cpp $(CPP_TESTS)/test_module.cpp $(CPP_TESTS)/test_helpers.h $(GTEST_LIB) -o $(CPP_TESTS)/test_module
# 	./$(CPP_TESTS)/test_module

# test_linear: $(CPP_TESTS)/test_linear.cpp $(CPP_SOURCE)/linear.h $(CPP_SOURCE)/linear.cpp
# 	$(CXX) $(CXXFLAGS) $(CPP_SOURCE)/linear.cpp $(CPP_TESTS)/test_linear.cpp $(GTEST_LIB) -o $(CPP_TESTS)/test_linear
# 	./$(CPP_TESTS)/test_linear

# test_utils: $(CPP_TESTS)/test_utils.cpp $(CPP_SOURCE)/utils.h $(CPP_SOURCE)/utils.cpp
# 	$(CXX) $(CXXFLAGS) $(CPP_SOURCE)/utils.cpp $(CPP_TESTS)/test_utils.cpp $(GTEST_LIB) -o $(CPP_TESTS)/test_utils
# 	./$(CPP_TESTS)/test_utils

# clean_torchcpp:
# 	rm -f $(CPP_TESTS)/test_module
# 	rm -f $(CPP_TESTS)/test_linear
# 	rm -f $(CPP_TESTS)/test_utils

# torchpp_lint:
# 	for file in $(CPP_SOURCE_FILES) $(CPP_TEST_FILES) ; do \
# 		clang-tidy $$file -- -std=c++20 -I/usr/include/eigen3 -I./ ; \
# 	done

# torchcpp_unit_tests: test_module test_utils test_linear clean_torchcpp

# torchcpp_tests: torchpp_lint torchcpp_unit_tests
