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
 * 
 * opencv version 4.10.0
 * https://www.youtube.com/watch?v=KecMlLUuiE4&list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI&index=2
 * 
 *
 * OpenCV 4.10 documentation.
 * https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
 * 
 * Nuclear Fuel Rods 
 * https://www.google.com/search?sca_esv=bcfcdd4518f8cd40&q=nuclear+fuel+rod+diagram&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWd8nbOJfsBGGB5IQQO6L3J7pRxUp2pI1mXV9fBsfh39JqJxzRlphkmT2MeVSzs3MQCUNkeUaVjRp3Bu8J5s0UhhW9p8XBQ4OgSxaZFuzMRIHm1YH1fcYQUvvrISxlCv9Km7C6ufW8zILU3_5Nhj-Vo4qKzb5wi7QUQvpxDIQTbaNzipXoYS-g5hS9YBSIbVVrGMsj-g&sa=X&ved=2ahUKEwjR_-jHnL-KAxVZCTQIHRpMBKQQtKgLegQIGRAB&biw=1605&bih=770&dpr=1#vhid=gI6KOEJ1Xt8XHM&vssid=mosaic
 */

/*
 * opencv_annotation -a=annotations.txt -i=positive/
 * press 'c' to accept a selection,
 * press 'd' to delete the latest selection,
 * press 'n' to proceed with next image,
 * press 'esc' to stop.  
 * opencv_createsamples -info annotations.txt -w 24 -h 24 -num 1000 -vec annotations.vec
 * opencv_traincascade -data cascade/ -vec annotations.vec -bg neg.txt -w 24 -h 24 -numPos 200 -numNeg 100 -numStages 20
 * 
 */

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

   cv::CascadeClassifier car_cascade;
   car_cascade.load("./cascade/xml/cascade.xml");
   cv::VideoCapture video_capture;
   video_capture = cv::VideoCapture("./17 December 2024.mp4");

   cv::Mat my_frame;
   while ( video_capture.read(my_frame))
   {
      std::vector< cv::Rect > cars_matrix;
      car_cascade.detectMultiScale( my_frame, cars_matrix);

      for( size_t i = 0; i < cars_matrix.size(); i++)
      {
         cv::Point center(cars_matrix[i].x + cars_matrix[i].width/2, cars_matrix[i].y + cars_matrix[i].height/2 );
         cv::ellipse ( my_frame, center, cv::Size( cars_matrix[i].width/2, cars_matrix[i].height/2 ), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4 );
      }
      cv::imshow("Capture", my_frame);
      cv::waitKey( 1 );
   }
   

   return 0;
}
