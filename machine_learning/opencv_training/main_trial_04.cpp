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
   cv::Mat my_image, mask_image;
   mask_image = cv::Mat::zeros( cv::Size( 1000, 1000),
                                CV_8UC3);

   std::cout << "Number of arguments: " << argc << std::endl;
  
   if( argc != 2 )
   {
      std::cout <<  "Please enter image file .. " << std::endl ;
      return -1;
   }

   my_image = cv::imread( argv[1], 
                          cv::IMREAD_GRAYSCALE );

   cv::namedWindow( "Mask",
                    0);

   int point1 = 100;
   int point2 = 200;
   int ration = 1;

   cv::createTrackbar( "Point1",
                        "Mask",
                        &point1,
                        500);

   cv::createTrackbar( "Point2",
                        "Mask",
                        &point2,
                        500);

   cv::Rect my_rectangle(100, 100, 100, 100);

   cv::rectangle( mask_image, 
                  my_rectangle,
                  cv::Scalar(0, 255, 255),
                  3, 
                  cv::LINE_4 );

   cv::rectangle( mask_image, 
                  cv::Point(100, 210),
                  cv::Point(200, 310), 
                  cv::Scalar(0, 255, 255),
                  3, 
                  cv::LINE_8 );

   cv::rectangle( mask_image, 
                  cv::Point(100, 320),
                  cv::Point(200, 420), 
                  cv::Scalar(0, 255, 255),
                  3, 
                  cv::LINE_AA );

   cv::arrowedLine( mask_image, 
                    cv::Point(point1, point2),
                    cv::Point(150, 150),
                    cv::Scalar(0, 255, 255),
                    cv::LINE_AA,
                    0,
                    (double)ration/10.0);

   cv::imshow( "Mask", 
                mask_image );

   cv::waitKey( 0 );

   return 0;
}
