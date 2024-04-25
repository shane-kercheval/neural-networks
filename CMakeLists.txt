cmake_minimum_required(VERSION 3.10)
# export a compilation database; this database contains all the commands and flags used to compile
# the project, which clang-tidy (makefile) can utilize to accurately analyze the code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(TorchCPP)

# -Wall enables most warning messages from the compiler
add_compile_options(-Wall)

# Create the core library
# PRIVATE indicates that these sources are only part of the implementation of the library and
# should not affect consumers of the library directly.
add_library(torchcpp_lib)
target_sources(torchcpp_lib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/torchcpp/linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/torchcpp/utils.cpp
)

# Set the C++ standard for the library, and means that torchcpp_lib requires the C++20 standard.
# PUBLIC means this requirement will also apply to any targets that are linked against this library.
target_compile_features(torchcpp_lib PUBLIC cxx_std_20)

# Include directories for the library: sets the directories that the compiler uses to search for
# header files. PUBLIC here affects both the library and any targets that link against it.
target_include_directories(torchcpp_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/torchcpp
)

# ensure torchcpp.h is also installed or made available
# sets properties for the target; in this case it designates torchcpp.h as a public header for the library.
set_target_properties(torchcpp_lib PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/torchcpp/torchcpp.h)

# find and link Eigen3 (version 3.3 or newer).
# REQUIRED means CMake will fail if Eigen3 isn't found.
# NO_MODULE means it uses Eigen3's config file rather than a Find Module.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    # Links the Eigen3 library to torchcpp_lib.
    # The PUBLIC specifier indicates that anyone linking against torchcpp_lib also links against Eigen3.
    target_link_libraries(torchcpp_lib PUBLIC Eigen3::Eigen)
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

# find Google Test
find_package(GTest REQUIRED)
if(NOT GTest_FOUND)
    message(FATAL_ERROR "Google Test not found. Please ensure it is installed.")
endif()

enable_testing()

# defines a function called create_test to create test executables
function(create_test target)
    add_executable(${target} ${CMAKE_CURRENT_SOURCE_DIR}/torchcpp_tests/${target}.cpp)
    # his line specifies that the newly created test executable should link against several libraries, include my own
    target_link_libraries(${target} torchcpp_lib GTest::GTest GTest::Main ${CMAKE_THREAD_LIBS_INIT})
    # This command adds include directories for the target.
    # PRIVATE means these directories are used only for compiling the specified target
    # This is, for example, including torchcpp_tests/test_helper.h in the test executables
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/torchcpp_tests)
    # registers the executable as a test with CTest, which is CMake's test driver
    add_test(NAME ${target} COMMAND ${target} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# Create test executables
create_test(test_module)
create_test(test_linear)
create_test(test_utils)

add_executable(torchcpp_mlp
    ${CMAKE_CURRENT_SOURCE_DIR}/multilayer_perceptron/torchcpp_mlp_mnist.cpp
)
target_link_libraries(torchcpp_mlp torchcpp_lib)
