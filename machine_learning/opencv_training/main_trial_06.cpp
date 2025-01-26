#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
// #include <opencv2/imgcodecs.hpp>

#include <iostream>

// #include <onnxruntime_cxx_api.h>

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
   cv::Mat _8uc1 = cv::Mat::zeros( cv::Size(3,4),                   
                                   CV_8UC1 ) ;

   cv::Mat _8uc2 = cv::Mat::zeros( cv::Size(3,4),                   
                                   CV_8SC1 ) ;

   _8uc2.at< char >(cv::Point(1, 1)) = 12;

   std::cout << _8uc1 
             << std::endl;

   std::cout << _8uc1.depth() 
             << std::endl;

   std::cout << _8uc1.type() 
             << std::endl;

   std::cout << _8uc1.size() 
             << std::endl;

   std::cout << _8uc1.cols 
             << std::endl;

   std::cout << _8uc1.rows 
             << std::endl;

   std::cout << _8uc1.channels() 
             << std::endl;

   std::cout << _8uc2 
             << std::endl;

   std::cout << _8uc2.depth() 
             << std::endl;

   std::cout << _8uc2.type() 
             << std::endl;

   std::cout << _8uc2.size() 
             << std::endl;

   std::cout << _8uc2.cols 
             << std::endl;

   std::cout << _8uc2.rows 
             << std::endl;

   std::cout << _8uc2.channels() 
             << std::endl;

   cv::namedWindow("8UC1", 0);
   cv::namedWindow("8UC2", 0);
   cv::imshow( "8UC1", 
               _8uc1);

   cv::imshow( "8UC2", 
               _8uc2);

   cv::Mat my_image;

   std::cout << "Number of arguments: " << argc << std::endl;
  
   if( argc != 2 )
   {
      std::cout <<  "Please enter image file .. " << std::endl ;
      return -1;
   }

   // cv::namedWindow("Output", CV_WINDOW_AUTOSIZE);
   cv::namedWindow("Output", CV_WINDOW_NORMAL);


   my_image = cv::imread( argv[1], 
                          cv::IMREAD_GRAYSCALE );
   
   cv::imshow( "Output",
                my_image);

   cv::waitKey( 0 );

   return 0;
}
