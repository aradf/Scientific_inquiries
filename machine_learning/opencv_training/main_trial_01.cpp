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
   cv::Mat bilaterial_image, gaussian_image, median_image, my_image;
   cv::Mat erode_matrix, dilate_matrix, element_kernel;
   
   std::cout << "Number of arguments: " << argc << std::endl;
  
   if( argc != 2 )
   {
      std::cout <<  "Please enter image file .. " << std::endl ;
      return -1;
   }
   
   my_image = cv::imread( argv[1], 
                          cv::IMREAD_COLOR );

   cv::imshow( "Hello World windows", 
                my_image );

   cv::waitKey(0);

   element_kernel = cv::getStructuringElement( cv::MORPH_RECT, 
                                               cv::Size( 15, 15), 
                                               cv::Point(-1,-1));
   cv::erode( my_image, 
              erode_matrix, 
              element_kernel, 
              cv::Point( -1, -1), 
              1);

   cv::dilate( my_image, 
               dilate_matrix, 
               element_kernel, 
               cv::Point(-1,-1), 1);

   cv::imshow( "Eroded windows", 
                erode_matrix );

   cv::waitKey(0);

   cv::imshow( "Dilated windows", 
                dilate_matrix );

   cv::waitKey(0);


   cv::bilateralFilter( my_image, 
                        bilaterial_image,
                        15,
                        95,
                        45 );
 
   cv::imshow( "Bilateral Image windows", 
                bilaterial_image );

   cv::waitKey(0);

   cv::GaussianBlur( my_image,
                     gaussian_image,
                     cv::Size( 15, 15),
                     0 );

   cv::imshow( "Gaussian Image windows", 
                gaussian_image );

   cv::waitKey(0);

   cv::medianBlur( my_image,
                   median_image,
                   3 );

   cv::imshow( "Median Image windows", 
                median_image );

   cv::waitKey(0);


   return 0;
}
