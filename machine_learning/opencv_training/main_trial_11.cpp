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

   {
      cv::Mat fuelrod_image = cv::imread( "./nuclear/gamma_released_nutron_activated.png");
      cv::Mat plenum_leftImage  = cv::imread("./nuclear/plenum_left.png");

      cv::Mat result_leftMatrix;   
      cv::matchTemplate( fuelrod_image, 
                        plenum_leftImage,
                        result_leftMatrix,
                        cv::TM_CCOEFF_NORMED);

      cv::imshow("Result", result_leftMatrix);
      cv::waitKey(0);

      // Get the best match position.
      double min_value = 0, max_value = 0;
      cv::Point min_location, max_location, match_location;
      minMaxLoc( result_leftMatrix,
               &min_value,
               &max_value,
               &min_location,
               &max_location);
      double threshold = 0.8;

      if (max_value >= threshold)
      {
         std::cout << "Best Match position " 
                  << max_location.x
                  << ","
                  << max_location.y
                  << std::endl;
   
         match_location = max_location;
         cv::rectangle( fuelrod_image, 
                        match_location, 
                        cv::Point( match_location.x + plenum_leftImage.cols , match_location.y + plenum_leftImage.rows ), 
                        cv::Scalar::all(0), 2, 8, 0 );
   
         cv::imshow("Result", fuelrod_image);
         cv::waitKey(0);
      }
   }

   {
      cv::Mat fuelrod_image = cv::imread( "./nuclear/gamma_released_nutron_activated.png");
      cv::Mat plenum_rightImage = cv::imread("./nuclear/plenum_right.png");

      cv::Mat result_rightMatrix;   
      cv::matchTemplate( fuelrod_image, 
                         plenum_rightImage,
                         result_rightMatrix,
                         cv::TM_CCOEFF_NORMED);

      cv::imshow("Result", result_rightMatrix);
      cv::waitKey(0);

      // Get the best match position.
      double min_value = 0, max_value = 0;
      cv::Point min_location, max_location, match_location;
      minMaxLoc( result_rightMatrix,
                 &min_value,
                 &max_value,
                 &min_location,
                 &max_location);
      double threshold = 0.8;

      if (max_value >= threshold)
      {
         std::cout << "Best Match position " 
                   << max_location.x
                   << ","
                   << max_location.y
                   << std::endl;
      
         match_location = max_location;
         cv::rectangle( fuelrod_image, 
                        match_location, 
                        cv::Point( match_location.x + plenum_rightImage.cols , match_location.y + plenum_rightImage.rows ), 
                        cv::Scalar::all(0), 2, 8, 0 );
   
         cv::imshow("Result", fuelrod_image);
         cv::waitKey(0);
      }
   }

   {
      cv::Mat fuelrod_image = cv::imread( "./nuclear/gamma_released_nutron_activated.png");
      cv::Mat activated_palletImage = cv::imread("./nuclear/activated_pallet.png");

      cv::Mat result_palletMatrix;   
      cv::matchTemplate( fuelrod_image, 
                         activated_palletImage,
                         result_palletMatrix,
                         cv::TM_CCOEFF_NORMED);

      // cv::matchTemplate( fuelrod_image, 
      //                    activated_palletImage,
      //                    result_palletMatrix,
      //                    cv::TM_SQDIFF_NORMED);

      cv::imshow("Result", result_palletMatrix);
      cv::waitKey(0);

      // Get the best match position.
      double min_value = 0, max_value = 0;
      cv::Point min_location, max_location, match_location;
      minMaxLoc( result_palletMatrix,
                 &min_value,
                 &max_value,
                 &min_location,
                 &max_location,
                 cv::Mat());
      double threshold = 0.49;
      std::vector< cv::Point > multiple_matches;
      
      for (int i=0; i < result_palletMatrix.rows; i++)
         for (int j=0; j < result_palletMatrix.cols; j++)
         {
            if (result_palletMatrix.at< float >(i, j) > threshold)
            {
               multiple_matches.push_back(cv::Point(j,i));
            }
         }

         // Sample rectangles (replace with your actual detections)
         // std::vector<cv::Rect> initialRects = 
         // {
         //    cv::Rect(10, 10, 50, 50),
         //    cv::Rect(15, 12, 52, 48),
         //    cv::Rect(200, 200, 60, 60),
         //    cv::Rect(205, 195, 55, 65),
         //    cv::Rect(300, 300, 40, 40)
         // };

         // // Group rectangles
         // std::vector<cv::Rect> groupedRects = initialRects;
         // cv::groupRectangles( groupedRects, 1, 0.5);


      std::cout << "Number of Matches found: " << multiple_matches.size() << std::endl;
      for( const auto& single_match : multiple_matches )
      {
         std::cout << single_match.x << " , " << single_match.y << std::endl;
         cv::Point match_location(single_match.x, single_match.y);
         cv::rectangle( fuelrod_image, 
                        match_location, 
                        cv::Point( match_location.x + activated_palletImage.cols , match_location.y + activated_palletImage.rows ), 
                        cv::Scalar::all(0), 2, 8, 0 );
      }
      cv::imshow("Result", fuelrod_image);
      cv::waitKey(0);

   }


   return 0;
}
