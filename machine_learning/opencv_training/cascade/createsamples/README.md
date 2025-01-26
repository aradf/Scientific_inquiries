# How to compile and link.

g++ createsamples.cpp utility.cpp -o opencv_createsamples -std=c++11 `pkg-config --cflags --libs opencv4`
