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

cv::Point2f updatePositionAlongRectangle(cv::Point2f current_position, cv::Point2f& direction, const cv::Rect& rect) 
{
    current_position += direction;

    if ( current_position.x <= rect.x || current_position.x >= rect.x + rect.width) 
    {
        direction.x = -direction.x;
    }
    if (current_position.y <= rect.y || current_position.y >= rect.y + rect.height) 
    {
        direction.y = -direction.y;
    }

    return current_position;
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
    * 
    * https://docs.opencv.org/4.10.0/dd/d6a/classcv_1_1KalmanFilter.html
    */
   cv::namedWindow("Kalman Filter Tracking", 0);
   cv::KalmanFilter kalman_filter(4, 2, 0);

   kalman_filter.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                              0, 1, 0, 1,
                                                              0, 0, 1, 0,
                                                              0, 0, 0, 1);

   kalman_filter.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,
                            0, 1, 0, 0);

   setIdentity( kalman_filter.processNoiseCov, cv::Scalar(1e-4) );
   setIdentity( kalman_filter.measurementNoiseCov, cv::Scalar(1e-1) );
   setIdentity( kalman_filter.errorCovPost, cv::Scalar(1) );

   kalman_filter.statePost = ( cv::Mat_< float >(4, 1) << 0, 0, 0, 0 );

   int width = 640;
   int height = 480;
   cv::Mat frame( height, width, CV_8UC3, cv::Scalar(0, 0, 0) );

   cv::Rect rectangle( 100, 100, 440, 280 );

    cv::Point2f true_position( rectangle.x + rectangle.width / 2, rectangle.y + rectangle.height / 2 );
    cv::Point2f direction( 5, 5 );

    cv::Point2f predicted_position;

   while (true) 
   {
        frame.setTo(cv::Scalar(0, 0, 0));

        true_position = updatePositionAlongRectangle( true_position, direction, rectangle );

        cv::Mat prediction = kalman_filter.predict();
        predicted_position = cv::Point2f( prediction.at< float >(0), prediction.at< float >(1) );

        cv::Mat measurement = (cv::Mat_< float >(2, 1) << true_position.x, true_position.y);
        kalman_filter.correct(measurement);

        cv::Mat future_state = kalman_filter.statePost.clone();

        for (int i = 0; i < 10; ++i) 
        { // predicting 10 steps ahead
            future_state = kalman_filter.transitionMatrix * future_state;
        }
        cv::Point future_predictPt( future_state.at< float >(0), future_state.at< float >(1));

        cv::rectangle( frame, rectangle, cv::Scalar(255, 0, 0), 2 ); // Rectangle in blue
        cv::circle( frame, true_position, 10, cv::Scalar(255, 255, 255), -1 ); // True position in white
        cv::circle( frame, predicted_position, 10, cv::Scalar(0, 255, 0), -1 ); // Predicted position in green
        cv::circle(frame, future_predictPt, 10, cv::Scalar(0, 0, 255), -1 ); // Predicted position in green

        cv::imshow("Kalman Filter Tracking", frame);

        if (cv::waitKey(0) == 'q')
           break;
    }

   return 0;
}
