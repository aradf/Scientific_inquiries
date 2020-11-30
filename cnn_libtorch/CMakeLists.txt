cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cnn_classify)

### Use the following line to run the cmake.
### cmake -DCMAKE_PREFIX_PATH=/home/faramarz/libtorch ..
### cmake --build . --config Release
### cmake --build . --config Debug
### cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/home/faramarz/libtorch ..
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cnn_mnist cnn_mnist_main.cpp)

add_executable(cnn_classify cnn_classify_main.cpp)

add_executable(cnn_vgg_16 cnn_vgg_16_main.cpp)

target_link_libraries(cnn_classify "${TORCH_LIBRARIES}")
target_link_libraries(cnn_mnist "${TORCH_LIBRARIES}")
target_link_libraries(cnn_vgg_16 "${TORCH_LIBRARIES}")

set_property(TARGET cnn_classify PROPERTY CXX_STANDARD 14)

