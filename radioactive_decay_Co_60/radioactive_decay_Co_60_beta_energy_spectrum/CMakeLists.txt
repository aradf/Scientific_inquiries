cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(RadioactiveDecaySimulation)

find_package(Geant4 REQUIRED ui_all vis_all)

include(${Geant4_USE_FILE})

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(BUILD_MODE Debug)
SET(CMAKE_CXX_FLAGS "-Wall")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")


file(GLOB sources ${PROJECT_SOURCE_DIR}/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/*.hh)

file(GLOB MACRO_FIELS "*.mac")
file(COPY ${MACRO_FIELS} DESTINATION ${PROJECT_BINARY_DIR})

file(GLOB DATA_FIELS "*.dat")
file(COPY ${DATA_FIELS} DESTINATION ${PROJECT_BINARY_DIR})

add_executable(decay_sim RadioactiveDecayMain.cc ${sources} ${headers})
target_link_libraries(decay_sim ${Geant4_LIBRARIES})

add_custom_target(RadioactiveDecaySimulation DEPENDS decay_sim)

