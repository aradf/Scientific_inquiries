cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project("Qt Basic" LANGUAGES CXX)

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
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt5 REQUIRED COMPONENTS Network)
find_package(Qt5 REQUIRED COMPONENTS Core)
find_package(Qt5 REQUIRED COMPONENTS Concurrent)

message(${PROJECT_SOURCE_DIR})
#----------------------------------------------------------------------------
# Setup include directory for this project
#
include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/mylib")
add_subdirectory (mylib)
set (MYLIB_LIB ${MYLIB_LIB} mylib)

message("MYLIB_LIB")
message(${MYLIB_LIB})

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.h)

add_executable(run_qtIODevice 
               ${PROJECT_SOURCE_DIR}/main.cpp 
               ${PROJECT_SOURCE_DIR}/src/first.cpp
               ${headers})

target_link_libraries(run_qtIODevice PRIVATE Qt5::Core Qt5::Widgets Qt5::Network Qt5::Concurrent ${MYLIB_LIB})

install(TARGETS run_qtIODevice DESTINATION ./build)




