#------------------------------
# Set up the Project
#
cmake_minimum_required(VERSION 3.16...3.21 FATAL_ERROR)
project(NUCE427LAB03)

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable
#
option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project
#
include(${Geant4_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/include)

message("------------------------")
message(${Geant4_USE_FILE})
message("------------------------")


#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(nuce427lab03 NUCE427Lab03_main.cc ${sources} ${headers})
target_link_libraries(nuce427lab03 ${Geant4_LIBRARIES})

#----------------------------------------------------------------------------
# Copy all scripts to the build directory, i.e. the directory in which we
# build NUCE427Lab03. This is so that we can run the executable directly 
# because it relies on these scripts being in the current working directory.
#
# set(NUCE427LAB03_SCRIPTS
#   debug.mac
#   init_vis.mac
#   run.mac
#   run1.mac
#   run2.mac
#   vis.mac
#   )

#   foreach(_script ${NUCE427LAB03_SCRIPTS})
#   configure_file(
#     ${PROJECT_SOURCE_DIR}/${_script}
#     ${PROJECT_BINARY_DIR}/${_script}
#     COPYONLY
#     )
# endforeach()

#----------------------------------------------------------------------------
# For internal Geant4 use - but has no effect if you build this
# example standalone
#
# add_custom_target(B3a DEPENDS exampleB3a)

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS nuce427lab03 DESTINATION bin )