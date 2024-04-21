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

torchpy_tests: linting unit_tests

####
# torchcpp - Pytorch-like library for C++
####
CPP_TESTS = ./torchcpp_tests
CPP_SOURCE = ./torchcpp
CXX = g++
CXXFLAGS = -std=c++20 -Wall -I./ -I/usr/include/gtest/ -pthread
# -Wall is to enable most warning messages from the compiler
# -I is to add the include directory to the compiler's search path; we add the current directory and the Google Test include directory
# -pthread is to enable POSIX threads which is required by gtest
GTEST_LIB = -lgtest_main -lgtest
# GTEST_LIB contains the linker flags to link against the Google Test libraries.
# Here it links against gtest_main and gtest. The gtest_main library provides a main function that
# runs all tests, so I don't need to define it.

test_linear: $(CPP_TESTS)/test_linear.cpp $(CPP_SOURCE)/linear.h
	$(CXX) $(CXXFLAGS) $(CPP_TESTS)/test_linear.cpp $(GTEST_LIB) -o $(CPP_TESTS)/test_linear
	./$(CPP_TESTS)/test_linear

test_utils: $(CPP_TESTS)/test_utils.cpp $(CPP_SOURCE)/torchcpp.h
	$(CXX) $(CXXFLAGS) $(CPP_TESTS)/test_utils.cpp $(GTEST_LIB) -o $(CPP_TESTS)/test_utils
	./$(CPP_TESTS)/test_utils

clean_torchcpp:
	rm -f $(CPP_TESTS)/test_linear
	rm -f $(CPP_TESTS)/test_utils

torchcpp_tests: test_linear clean_torchcpp
