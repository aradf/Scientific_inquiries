cmake_minimum_required(VERSION 3.7)
#project(ann)
project(svm LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++14 -g -O0 -fopenmp -pthread -Wfatal-errors")

add_executable(run_svm main.cpp penhaun_svm.cpp libsvm/svm.cpp)
  
#target_link_libraries(train -pthread -fopenmp)
