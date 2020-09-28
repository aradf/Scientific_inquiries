cmake_minimum_required(VERSION 3.7)
#project(ann)
project(ann LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++14 -g -O0 -fopenmp -pthread -Wfatal-errors")

add_executable(main 
    src/main.cpp 
    src/common.cpp 
    src/data.cpp 
    src/data_handler.cpp 
    src/layer.cpp 
    src/network.cpp 
    src/neuron.cpp 
    src/helper_function.cpp
    src/training_handler.cpp)
  
#target_link_libraries(train -pthread -fopenmp)
