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

int main( int argc, char** argv )
{
   std::cout << "OpenCV version: " << CV_VERSION << std::endl;
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

   double min_blue  = 21;
   double min_green = 222;
   double min_red   = 70;

   double max_blue  = 176;
   double max_green = 255;
   double max_red   = 255;

   // Defining object for reading video from camera
   cv::VideoCapture camera(0);

   // Defining loop for catching frames   
   while(true)
   {
      // Capture frame-by-frame from camera
      cv::Mat frame_BGR;
      camera.read( frame_BGR );

      // Converting current frame to HSV
      cv::Mat frame_HSV;
      cv::cvtColor(frame_BGR, frame_HSV, cv::COLOR_BGR2HSV);

      // Implementing Mask with founded colours from Track Bars to HSV Image
      cv::Mat Mask;
      cv:inRange( frame_HSV, 
                  cv::Scalar(min_blue, min_green, min_red),
                  cv::Scalar(max_blue, max_green, max_red),
                  Mask );

      // Showing current frame with implemented Mask
      // Giving name to the window with Mask
      // Specify that window is resizable.
      cv::namedWindow("Binary frame with Mask", cv::WINDOW_NORMAL);
      cv::imshow("Binary frame with Mask", Mask);
      
      std::vector< std::vector < cv::Point > > contours;
      // Checking if OpenCv version is 4 is used.
      // https://docs.opencv.org/4.10.0/df/d0d/tutorial_find_contours.html
      // cv::findContours(Mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
      cv::findContours(Mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

      std::vector< std::vector < cv::Point > > contours_poly( contours.size() );
      std::vector< cv::Rect > boundRect( contours.size() );
      //std::vector< cv::Point2f > centers( contours.size() );
      //std::vector< float > radius( contours.size() );

      // Getting rectangle coordinates and spatial size from biggest Contour
      // Function cv2.boundingRect() is used to get an approximate rectangle
      // around the region of interest in the binary image after Contour was found

      double previous_area=0;
      int index_maxarea = 0;
      for (size_t iCnt=0; iCnt < contours.size(); iCnt++)
      {
         double area = cv::contourArea(contours[iCnt]);
         if (area > previous_area)
         {
            index_maxarea = iCnt;
         }
      }

      if (index_maxarea == 0)
         continue;

      cv::approxPolyDP( contours[index_maxarea], contours_poly[index_maxarea], 3, true );
      boundRect[index_maxarea] = cv::boundingRect( contours_poly[index_maxarea] );
      cv::Rect bound_rect = boundRect[index_maxarea];
      int x_min, y_min, box_width, box_height;
      x_min = boundRect[index_maxarea].x;
      y_min = boundRect[index_maxarea].y;
      box_width = boundRect[index_maxarea].width;
      box_height = boundRect[index_maxarea].height;


      // Drawing Bounding Box on the current BGR frame
      cv::rectangle( frame_BGR, 
                     cv::Point(x_min - 15, y_min - 15),
                     cv::Point(x_min + box_width + 15, y_min + box_height + 15),
                     (0, 255, 0), 
                     3);

      // Putting text with Label on the current BGR frame
      cv::putText( frame_BGR, 
                   "Detected Object", 
                   cv::Point(x_min - 5, y_min - 25),
                   cv::FONT_HERSHEY_SIMPLEX, 
                   1.0, 
                   (0, 255, 0), 
                   2);

      cv::namedWindow("Detected Object", cv::WINDOW_NORMAL);
      cv::imshow( "Detected Object", frame_BGR );
      if ( cv::waitKey(1) == 'q')
        break;
   }

   // Destroying all opened windows
   cv::destroyAllWindows();
   return 0;
}
