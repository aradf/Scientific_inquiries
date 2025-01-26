#include <fstream>

#include <opencv2/opencv.hpp>
// #include <opencv2/core/utility.hpp>

#include "include/help.hpp"

/*
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * pkg-config --modversion opencv4
 * g++ main.cpp -o output `pkg-config --cflags --libs opencv4`
 * g++ main.cpp -g -o ./run_onnx -I/usr/local/include/opencv4 -L/usr/local/lib `pkg-config --cflags --libs opencv4` -std=c++14
 * g++ main.cpp -g -o ./run_onnx -lX11 -Iinclude/ -Llib/ -lonnxruntime -Wl,-rpath=./lib
 * g++ main.cpp -g -o ./run_onnx -I/usr/local/include/opencv4 -L/usr/local/lib `pkg-config --cflags --libs opencv4` -std=c++14 -lX11 -Iinclude/ -Llib/ -lonnxruntime -Wl,-rpath=./lib
 * https://pub.towardsai.net/detecting-objects-with-yolov5-opencv-python-and-c-c7cf13d1483c
 * https://github.com/doleron/yolov5-opencv-cpp-python
 */

int main(int argc, char **argv)
{
   const cv::String keys =
        "{help h usage ? |      | print this message   }"
        "{@image | | Path to the input image}"
        "{blur b | 5 | Blur kernel size (must be odd)}";

   cv::CommandLineParser parser(argc, argv, keys);
   parser.about("Yolov5_onnx Application v1.0.0");

   if( parser.has("help"))
   {
      parser.printMessage();
      return 0;
   }

   std::cout << "Initialization: " 
             << argv[1] 
             << " " 
             << argv[2] 
             << std::endl;

   std::vector< std::string > class_list = YOLOV5::loadyolov5_classList();
   cv::Mat single_frame;

#ifdef BEST_ONNX
   cv::VideoCapture capture("forest-road.mp4");
#else
   cv::VideoCapture capture("sample.mp4");
#endif
    
   if (!capture.isOpened())
   {
      std::cerr << "Error opening video file\n";
      return -1;
   }

   bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

   cv::dnn::Net yolov5_network;
   YOLOV5::load_yolvo5Network( yolov5_network, is_cuda);

   auto start = std::chrono::high_resolution_clock::now();
   int frame_count = 0;
   float fps = -1;
   int total_frames = 0;

   while (true)
   {
      capture.read( single_frame );
      if ( single_frame.empty() )
      {
         std::cout << "End of stream\n";
         break;
      }

      std::vector< YOLOV5::Prediction > detected_output;
      inference_yolov5( single_frame, 
                        yolov5_network, 
                        detected_output, 
                        class_list );

      frame_count++;
      total_frames++;

      int predictions = detected_output.size();

      for (int i = 0; i < predictions; ++i)
      {
          auto single_prediction = detected_output[i];
          auto box = single_prediction.box;
          auto classId = single_prediction.class_id;
          const auto color = YOLOV5::colors[ classId % YOLOV5::colors.size() ];
          cv::rectangle( single_frame, 
                         box, 
                         color, 
                         3 );

          cv::rectangle( single_frame, 
                         cv::Point(box.x, box.y - 20), 
                         cv::Point(box.x + box.width, box.y), 
                         color, 
                         cv::FILLED);

          cv::putText( single_frame, 
                       class_list[classId].c_str(), 
                       cv::Point(box.x, box.y - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       cv::Scalar(0, 0, 0) );
      }

      if ( frame_count >= 30 )
      {
          auto end = std::chrono::high_resolution_clock::now();
          fps = frame_count * 1000.0 / std::chrono::duration_cast< std::chrono::milliseconds >(end - start).count();

          frame_count = 0;
          start = std::chrono::high_resolution_clock::now();
      }

      if ( fps > 0 )
      {
          std::ostringstream fps_label;
          fps_label << std::fixed 
                    << std::setprecision(2);

          fps_label << "FPS: " 
                    << fps;

          std::string fps_label_str = fps_label.str();

          cv::putText( single_frame, 
                       fps_label_str.c_str(), 
                       cv::Point(10, 25), 
                       cv::FONT_HERSHEY_SIMPLEX, 
                       1, 
                       cv::Scalar(0, 0, 255), 
                       2 );
      }

      cv::imshow( "output", 
                  single_frame );

      if ( cv::waitKey(1) != -1 )
      {
          capture.release();
          std::cout << "finished by user ..." 
                    << std::endl;
          break;
      }
    }

    std::cout << "Total frames: " << total_frames << "\n";

    return 0;
}