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
   /*
    * the OpenCV namespace has a Mat_ container holding floats.
    */
   cv::Mat_< float > custom_matrix( 3, 3);
   cv::Mat_< float > kernel_matrix( 3, 3);

   std::cout << "Number of arguments: " << argc << std::endl;
  
   if( argc != 2 )
   {
      std::cout <<  "Please enter image file .. " << std::endl ;
      return -1;
   }

   custom_matrix = cv::imread( argv[1], 
                               cv::IMREAD_GRAYSCALE );

   kernel_matrix << -1, 1, -1, 1, 0, 1, -1, 1, -1;

   cv::namedWindow( "Custom",
                    cv::WINDOW_NORMAL);
   cv::namedWindow( "Kernel",
                    cv::WINDOW_NORMAL);
   cv::namedWindow( "Filter2D",
                    cv::WINDOW_NORMAL);

   cv::Mat custom_twoDim, kernel_twoDim, filter_twoDim, filter_twoDim_twoDim;
   cv::filter2D( custom_matrix, 
                 filter_twoDim, 
                 -1, 
                 kernel_matrix, 
                 cv::Point( -1, -1) );

   custom_matrix.convertTo( custom_twoDim, 
                            CV_8UC1 );
   kernel_matrix.convertTo( kernel_twoDim, 
                            CV_8UC1 );
   filter_twoDim.convertTo( filter_twoDim_twoDim, 
                            CV_8UC1 );

   cv::imshow( "Custom", 
                custom_twoDim );
   cv::imshow( "Kernel", 
                kernel_twoDim );
   cv::imshow( "Filter2D", 
                filter_twoDim_twoDim );

   cv::waitKey( 0 );

   return 0;
}
