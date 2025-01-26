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

   cv::VideoCapture video_capture(0);
   cv::Mat knn_frame;

   cv::namedWindow("Input", 0);
   cv::namedWindow("Output", 0);

   /*
    * K-nearest neighbours - based background/foreground segmentation.
    * https://docs.opencv.org/4.x/de/de1/group__video__motion.html
    */
   // cv::Ptr< cv::BackgroundSubtractorKNN > knearest_neighbour;
   // knearest_neighbour = cv::createBackgroundSubtractorKNN( 500, 
   //                                                         400, 
   //                                                         false);

   cv::Ptr< cv::BackgroundSubtractorMOG2 > mog2;
   mog2 = cv::createBackgroundSubtractorMOG2( 500, 
                                              400, 
                                              false);


   while (1)
   {
      video_capture >> knn_frame;
      cv::Mat output_frame;
      // knearest_neighbour->apply( knn_frame, 
      //                            output_frame );

      mog2->apply( knn_frame, 
                   output_frame );


      cv::imshow( "Input", 
                  knn_frame );
      cv::imshow( "Ooutput", 
                  output_frame );

      cv::waitKey( 1 );

   }

   return 0;
}
