cmake_minimum_required(VERSION 3.16...3.21 FATAL_ERROR)
project( DetectingObject )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
message("${OpenCV_INCLUDE_DIRS}")
#----------------------------------------------------------------------------
# Path to ONNX Runtime installation directory.
find_package( onnxruntime REQUIRED )
include_directories(${onnxruntime_INCLUDE_DIRS})
add_executable( DetectingObject detectingobject_main.cpp)
target_link_libraries( DetectingObject ${OpenCV_LIBS} ${onnxruntime_LIBRARY})
