cmake_minimum_required(VERSION 3.13)
project(Yolov5_onnx)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Path to ONNX Runtime installation directory.
find_package( onnxruntime REQUIRED )
include_directories(${onnxruntime_INCLUDE_DIRS})
add_executable(Yolov5_onnx main.cpp src/help.cpp)
target_link_libraries(Yolov5_onnx ${OpenCV_LIBS} ${onnxruntime_LIBRARY})

