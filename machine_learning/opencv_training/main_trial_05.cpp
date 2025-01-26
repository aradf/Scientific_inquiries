// #include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <iostream>

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
   cv::Mat my_frame = cv::Mat::zeros( cv::Size(500, 500), CV_8UC3);

   std::cout << "Number of arguments: " << argc << std::endl;
  
   if( argc != 2 )
   {
      std::cout <<  "Please enter image file .. " << std::endl ;
      return -1;
   }

   // cv::namedWindow("Output", CV_WINDOW_AUTOSIZE);
   cv::namedWindow("Output", CV_WINDOW_NORMAL);

   cv::putText( my_frame, 
                "Hello",
                cv::Point( 100, 100),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 255),
                2);

   int bL = 0;
   cv::Size my_size;
   my_size = cv::getTextSize( "Hello",
                               cv::FONT_HERSHEY_SIMPLEX,
                               1.0, 
                               2, 
                               &bL);

   std::cout << "Text size: " 
             << my_size 
             << std::endl;

   cv::Mat my_image;

   my_image = cv::imread( argv[1], 
                          cv::IMREAD_GRAYSCALE );

   // cv::addText( my_frame,
   //              "Hello",
   //              cv::Point(100, 200),
   //              cv::FONT_HERSHEY_SIMPLEX,
   //              "Times",
   //              30,
   //              cv::Scalar(0, 255, 255),
   //              cv::QT_FONT_NORMAL);

   cv::imshow( "Output", 
               my_frame );
   
   cv::waitKey( 0 );

   return 0;
}
