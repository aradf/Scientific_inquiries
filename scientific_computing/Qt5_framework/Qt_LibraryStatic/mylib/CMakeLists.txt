cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project("mylib" LANGUAGES CXX)

# message("Debugged this far ...")
# Find includes corresponding build directores.
set(CMAKE_INCLUDE_CURRENT_DIR on)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC on)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -Wfatal-errors")

#----------------------------------------------------------------------------
# Find Qt packges
#
find_package(Qt5 REQUIRED COMPONENTS Core)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.h)

add_library(mylib STATIC ${PROJECT_SOURCE_DIR}/src/mylib.cpp ${headers})

target_link_libraries(mylib PRIVATE Qt5::Core)
