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
 * https://www.youtube.com/watch?v=GyqENnu7cE8&list=PLUTbi0GOQwghR9db9p6yHqwvzc989q_mu&index=1
 * 
 *
 * OpenCV 4.10 documentation.
 * https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
 * 
 * cascade clasifier model training.
 * https://www.youtube.com/watch?v=XrCAvs9AePM
 * https://www.youtube.com/watch?v=Yq9JKSLke3Q&list=PLUTbi0GOQwghR9db9p6yHqwvzc989q_mu&index=149
 * 
 * Nuclear Fuel Rods 
 * https://www.google.com/search?sca_esv=bcfcdd4518f8cd40&q=nuclear+fuel+rod+diagram&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWd8nbOJfsBGGB5IQQO6L3J7pRxUp2pI1mXV9fBsfh39JqJxzRlphkmT2MeVSzs3MQCUNkeUaVjRp3Bu8J5s0UhhW9p8XBQ4OgSxaZFuzMRIHm1YH1fcYQUvvrISxlCv9Km7C6ufW8zILU3_5Nhj-Vo4qKzb5wi7QUQvpxDIQTbaNzipXoYS-g5hS9YBSIbVVrGMsj-g&sa=X&ved=2ahUKEwjR_-jHnL-KAxVZCTQIHRpMBKQQtKgLegQIGRAB&biw=1605&bih=770&dpr=1#vhid=gI6KOEJ1Xt8XHM&vssid=mosaic
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

   /*
    * 
    */
   cv::VideoCapture capture_video( 0 );
   cv::namedWindow( "Output", 0 );
   cv::CascadeClassifier my_cascade;
   my_cascade.load("./haarcascade_eye.xml");

   cv::Mat current_frame, gray_image;
   while ( 1 )
   {
       capture_video >> current_frame;
       cv::cvtColor( current_frame, gray_image, cv::COLOR_BGR2GRAY );

       std::vector< cv::Rect > eyes;
       my_cascade.detectMultiScale( gray_image, 
                                    eyes,
                                    1.1,
                                    3,
                                    0,
                                    cv::Size(),
                                    cv::Size() );

       for (size_t i = 0; i < eyes.size() ; i++)
       {
         cv::rectangle( current_frame, eyes[i],
                        cv::Scalar(255,0,0), 
                        2 );

         cv::imshow("Output", current_frame);

         cv::waitKeyEx(1);

       }
   }

   return 0;
}
