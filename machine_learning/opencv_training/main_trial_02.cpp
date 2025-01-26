// #include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <onnxruntime_cxx_api.h>

#include <iostream>

/*
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * g++ main.cpp -o output -std=c++11 `pkg-config --cflags --libs opencv4`
 * pkg-config --modversion opencv4
 * https://docs.opencv.org/4.x/index.html
 * opencv version 4.10.0
 * https://www.youtube.com/watch?v=GyqENnu7cE8&list=PLUTbi0GOQwghR9db9p6yHqwvzc989q_mu&index=1
 */

int main( int argc, char** argv )
{
   cv::Mat my_image, output_image;   
   std::cout << "Number of arguments: " << argc << std::endl;
  
   if( argc != 2 )
   {
      std::cout <<  "Please enter image file .. " << std::endl ;
      return -1;
   }
   
   my_image = cv::imread( argv[1], 
                          cv::IMREAD_GRAYSCALE );

   cv::namedWindow("Input", cv::WINDOW_NORMAL);
   cv::namedWindow("Output", cv::WINDOW_NORMAL);

   int key_stroke;
   int dx = 1, dy = 0, sobel_kernelSize = 3, sclae_factor = 1, delta_value = 1;
   while( 1 )
   {
      cv::Sobel( my_image, 
                 output_image,
                 CV_8U, 
                 dx, 
                 dy, 
                 sobel_kernelSize,
                 sclae_factor,
                 delta_value );

      key_stroke = cv::waitKey ( 1 );

      cv::imshow( "Input", 
                  my_image );

      key_stroke = cv::waitKey ( 0 );

      cv::imshow( "Output", 
                  output_image );

      if ((char)key_stroke == 'z')
         break;
   }

   return 0;
}
