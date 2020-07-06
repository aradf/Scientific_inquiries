cmake_minimum_required (VERSION 2.811 FATAL_ERROR) #need > 2.8 for QT support
project (ArtNeuralNetWork LANGUAGES CXX)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g -O0 -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -Wfatal-errors")

# for Linux
link_directories("/opt/qt512/lib")

# find pthreads
find_package(Threads)

# Tell CMake to create the daqSvr_Multi executable
add_executable(ArtNeuralNetWork main.cpp Cmnist_loader.cpp CNetwork.cpp)

# Use the Widgets module from Qt 5.
target_link_libraries(ArtNeuralNetWork ${CMAKE_THREAD_LIBS_INIT})



