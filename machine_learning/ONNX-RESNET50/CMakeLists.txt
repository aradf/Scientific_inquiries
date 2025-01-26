cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Path to ONNX Runtime installation directory.
find_package( onnxruntime REQUIRED )
include_directories(${onnxruntime_INCLUDE_DIRS})
add_executable( DisplayImage OnnxRuntimeResNet.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} ${onnxruntime_LIBRARY})
