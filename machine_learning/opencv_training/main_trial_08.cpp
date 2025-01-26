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
 * HighwWay
 * https://www.youtube.com/watch?v=KBsqQez-O4w
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
    * https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
    */
   cv::Ptr< cv::DISOpticalFlow > dis_opticalFlow = cv::DISOpticalFlow::create();
   cv::Mat frame, previous_frame, flow;
   cv::namedWindow( "Output", 0 );
   cv::VideoCapture video_capture("./4K Video of Highway Traffic!.mp4");

   // std::cout << dis_opticalFlow->getFinestScale() << std::endl;
   // std::cout << dis_opticalFlow->getGradientDescentIterations() << std::endl;
   // std::cout << dis_opticalFlow->getPatchSize() << std::endl;
   // std::cout << dis_opticalFlow->getPatchStride() << std::endl;
   std::cout << dis_opticalFlow->getUseMeanNormalization() << std::endl;
   std::cout << dis_opticalFlow->getUseSpatialPropagation() << std::endl;
   std::cout << dis_opticalFlow->getVariationalRefinementAlpha() << std::endl;

   // dis_opticalFlow->setFinestScale(10);
   // dis_opticalFlow->setGradientDescentIterations(100);
   // dis_opticalFlow->setPatchStride(1);
   int counter = 0;
   while ( 1 )
   {
      video_capture >> frame;
      cv::cvtColor( frame, frame, cv::COLOR_BGR2GRAY);
      counter++;
      if ( !previous_frame.empty() && counter % 15 == 0)
      {
         dis_opticalFlow->calc( previous_frame, frame, flow );
         for( int y=0; y<frame.rows; y+=15)
         {
            for( int x=0; x<frame.cols; x+=15)
            {
               cv::Point2f flowat_point = flow.at< cv::Point2f > (y ,x);
               cv::line( frame, 
                         cv::Point(x,y),
                         cv::Point( cvRound( x + flowat_point.x ),
                                    cvRound( y + flowat_point.y)),
                         cv::Scalar(0,0,0),
                         2);
               cv::circle( frame,
                           cv::Point(x,y), 
                           1, 
                           cv::Scalar(0,255,255),
                           1);

            }
         }
      }
      if (counter % 14 == 0)
         previous_frame = frame.clone();

      cv::imshow( "Output", 
                  frame );
      cv::waitKey( 1 );
   }


   return 0;
}
