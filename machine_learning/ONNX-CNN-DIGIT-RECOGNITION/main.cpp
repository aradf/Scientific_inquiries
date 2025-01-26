// #include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include "Helpers.cpp"

/*
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * pkg-config --modversion opencv4
 * g++ main.cpp -o output `pkg-config --cflags --libs opencv4`
 * g++ main.cpp -g -o ./run_onnx -I/usr/local/include/opencv4 -L/usr/local/lib `pkg-config --cflags --libs opencv4` -std=c++14
 * g++ main.cpp -g -o ./run_onnx -lX11 -Iinclude/ -Llib/ -lonnxruntime -Wl,-rpath=./lib
 * g++ main.cpp -g -o ./run_onnx -I/usr/local/include/opencv4 -L/usr/local/lib `pkg-config --cflags --libs opencv4` -std=c++14 -lX11 -Iinclude/ -Llib/ -lonnxruntime -Wl,-rpath=./lib
 */

/*
 * https://github.com/cassiebreviu/cpp-onnxruntime-resnet-console-app/blob/main/OnnxRuntimeResNet/OnnxRuntimeResNet.cpp
 */

int main( int argc, char** argv )
{
   std::cout << "Number of arguments: " << argc << std::endl;
  
   if( argc != 2 )
   {
      std::cout <<  "Please enter image file .. " << std::endl ;
      return -1;
   }
   Ort::Env env;
   Ort::RunOptions run_options;
   Ort::Session onnx_session(nullptr);

   constexpr int64_t number_channels = 3;
   constexpr int64_t width = 64;
   constexpr int64_t height = 64;
   constexpr int64_t number_classes = 10;
   constexpr int64_t number_inputElements = number_channels * height * width;

   // const std::string image_file = "./sample.jpg";
   const std::string image_file = argv[1];
   const std::string label_file = "./imagenet_classes.txt";
   auto model_path = "./cnn_onnx.keras.onnx";

   // load labels
   std::vector < std::string > labels = loadLabels( label_file );
   if ( labels.empty() )
   {
      std::cout << "Failed to load labels: " << label_file << std::endl;
      return 1;
   }

   // load image
   const std::vector< float > image_vector = loadImage( image_file, 64, 64 );
   if (image_vector.empty()) 
   {
      std::cout << "Failed to load image: " << image_file << std::endl;
      return 1;
   }

   if ( image_vector.size() != number_inputElements ) 
   {
      std::cout << "Invalid image format. Must be 224x224 RGB image." << std::endl;
      return 1;
   }

   // create session
   Ort::SessionOptions ort_sessionOptions;
   onnx_session = Ort::Session ( env,
                                 model_path,
                                 ort_sessionOptions );  

   // Define shape.
   const std::array< int64_t, 4 > input_shape = { 1, height, width, number_channels };
   const std::array< int64_t, 2 > output_shape = { 1, number_classes };

   // Define array
   std::array< float, number_inputElements > input_array;
   std::array< float, number_classes > results_array;

   // Define Tensor
   auto memory_info  = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
   auto input_tensor = Ort::Value::CreateTensor< float >( memory_info, 
                                                          input_array.data(), 
                                                          input_array.size(), 
                                                          input_shape.data(), 
                                                          input_shape.size() );
   auto output_tensor = Ort::Value::CreateTensor< float >( memory_info, 
                                                           results_array.data(), 
                                                           results_array.size(), 
                                                           output_shape.data(), 
                                                           output_shape.size());
   // copy imaget data to input_array
   std::copy( image_vector.begin(), image_vector.end(), input_array.begin() );

   // define names
   Ort::AllocatorWithDefaultOptions ort_allocated;
   Ort::AllocatedStringPtr input_name  = onnx_session.GetInputNameAllocated(0, ort_allocated);
   Ort::AllocatedStringPtr output_name = onnx_session.GetOutputNameAllocated(0, ort_allocated);
   const std::array< const char*, 1 > input_names = { input_name.get()};
   const std::array< const char*, 1 > output_names = { output_name.get()};
   input_name.release();
   output_name.release();   

   // run inference
   try 
   {
      onnx_session.Run( run_options, 
                        input_names.data(), 
                        &input_tensor, 
                        1, 
                        output_names.data(), 
                        &output_tensor, 
                        1);
   }
   catch (Ort::Exception& e) 
   {
      std::cout << e.what() << std::endl;
      return 1;
   }

   // sort results
   std::vector< std::pair<size_t, float> > index_valuePairs;
   for (size_t i = 0; i < results_array.size(); ++i) 
   {
      index_valuePairs.emplace_back(i, results_array[i]);
      std::cout << "Result Array " << i << " " << results_array[i] << std::endl;
   }

   std::sort( index_valuePairs.begin(), 
              index_valuePairs.end(), 
              [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

   // show Top5
   for (size_t i = 0; i < 9; ++i) 
   {
      const auto& result = index_valuePairs[i];
      std::cout << i + 1 << ": " << labels[result.first] << " " << result.second << std::endl;
   }

   return 0;
}
