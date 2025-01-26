/*
 * Simple Object Detection by thresholding with mask 
 * File: findingmask_main.cpp
 */

// Importing needed library
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <iostream>

// #include <onnxruntime_cxx_api.h>

/*
 * Convenient way for choosing right Colour Mask to Detect needed Object
 * Algorithm:
 * Reading RGB image --> Converting to HSV --> Getting Mask
 * Result:
 * min_blue, min_green, min_red = 21, 222, 70
 * max_blue, max_green, max_red = 176, 255, 255
 */

/*
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * g++ main.cpp -o output -std=c++11 `pkg-config --cflags --libs opencv4`
 * pkg-config --modversion opencv4
 * https://docs.opencv.org/4.x/index.html
 * 
 * opencv version 4.10.0
 * https://www.youtube.com/watch?v=KecMlLUuiE4&list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI&index=2
 * https://www.youtube.com/watch?v=mRhQmRm_egc
 * 
 */

/*
 * Preparing Track Bars - return void.
 */
void on_trackbar( int, void* )
{
}


int main( int argc, char** argv )
{
   std::cout << "Number of arguments: " << argc << std::endl;
  
   if( argc != 2 )
   {
      std::cout <<  "Please enter image file .. " << std::endl ;
      return -1;
   }

   cv::Mat my_image = cv::imread(argv[1]);
   cv::namedWindow("Windows", 0);
   cv::imshow("Windows", my_image);
   cv::waitKey( 0 );

   /*
    * Giving name to the window with Track Bars and specifying that window is resizable
    * Defining Track Bars for convenient process of choosing colours
    * https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html
    */
   cv::namedWindow( "Track Bars", 
                    cv::WINDOW_NORMAL );
   
   // For minimum range
   const int slider_max = 255;

   cv::createTrackbar("min_blue", "Track Bars", nullptr, slider_max, on_trackbar);
   cv::createTrackbar("min_green", "Track Bars", nullptr, slider_max, on_trackbar);
   cv::createTrackbar("min_red", "Track Bars", nullptr, slider_max, on_trackbar);

   // For maximum range
   cv::createTrackbar("max_blue", "Track Bars", nullptr, slider_max, on_trackbar);
   cv::createTrackbar("max_green", "Track Bars", nullptr, slider_max, on_trackbar);
   cv::createTrackbar("max_red", "Track Bars", nullptr, slider_max, on_trackbar);


   // cv::waitKey( 0 );

   // Reading image with OpenCV library. In this way image is opened already as an array
   // WARNING! OpenCV by default reads images in BGR format
   cv::Mat image_BGR = cv::imread("objects-to-detect.jpg");

   // Resizing image in order to use smaller windows
   cv::resize(image_BGR, image_BGR, cv::Size(600, 426), 0, 0, cv::INTER_LINEAR_EXACT);

   cv::namedWindow( "Original Image", cv::WINDOW_NORMAL );
   cv::imshow( "Original Image" , image_BGR);

   // cv::waitKey( 0 );

   // Converting original Image to Hue, Saturation and Value format (HSV)
   cv::Mat image_HSV;
   cv::cvtColor( image_BGR, image_HSV, cv::COLOR_BGR2HSV, 0);

   cv::namedWindow("HSV Image", cv::WINDOW_NORMAL);
   cv::imshow("HSV Image", image_HSV);

   // cv::waitKey( 0 );

   double min_blue = 0.0;
   double min_green = 0.0;
   double min_red = 0.0;

   // For maximum range
   double max_blue = 0.0;
   double max_green = 0.0;
   double max_red = 0.0;


   // Defining loop for choosing right colours for he Mask.
   while( true )
   {
      // Defining variables for saving values of the Track Bars 
      // For minimum range
      min_blue = cv::getTrackbarPos("min_blue", "Track Bars");
      min_green = cv::getTrackbarPos("min_green", "Track Bars");
      min_red = cv::getTrackbarPos("min_red", "Track Bars");

      // For maximum range
      max_blue = cv::getTrackbarPos("max_blue", "Track Bars");
      max_green = cv::getTrackbarPos("max_green", "Track Bars");
      max_red = cv::getTrackbarPos("max_red", "Track Bars");

      // Implementing Mask with chosen colours from Track Bars to HSV Image
      // Defining lower bounds and upper bounds for thresholding
      cv::Mat Mask;
      cv:inRange( image_HSV, 
                  cv::Scalar(min_blue, min_green, min_red),
                  cv::Scalar(max_blue, max_green, max_red),
                  Mask );

      // Showing Binary Image with implemented Mask
      // Giving name to the window with Mask
      // And specifying that window is resizable
      cv::namedWindow( "Binary Image with Mask", cv::WINDOW_NORMAL );
      cv::imshow( "Binary Image with Mask", Mask );

      // Breaking the loop if 'q' is pressed
      if ( cv::waitKey(1) == 'q') 
         break;

   }

   // Destroying all opened windows
   cv::destroyAllWindows();

   // Printing final chosen Mask numbers
   std::cout << "min_blue, min_green, min_red" << std::endl;
   std::cout << min_blue << "   " << min_green << "   " << min_red << std::endl;
   std::cout << "max_blue, max_green, max_red" << std::endl;
   std::cout << max_blue << "   " << max_green << "   " << max_red << std::endl;

   return 0;
}
