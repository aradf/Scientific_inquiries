/*
 * Training YOLO v3 for Objects Detection with Custom Data
 * Objects Detection in Real Time with YOLO v3 and OpenCV
 * File: yolo3_camera_main.cpp
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>
#include <chrono>

/*
 * https://www.google.com/search?client=ubuntu&channel=fs&q=opencv+c%2B%2B+yolo4
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

    std::ifstream label_file("yolo-coco-data/coco.names");
    std::vector< std::string > labels;

    if( label_file.is_open())
    {
        std::string line;
        while( std::getline(label_file, line))
        {
            labels.push_back(line);
        }
        label_file.close();
    }
    else
    {
        std::cerr << "Error opening label file ... "  << std::endl;
        return -1;
    }

    for (const auto &label : labels)
    {
        std::cout << label << ", ";
    }
    std::cout << std::endl;

    // Load YOLOv4 model
    std::string darknetModel = "yolo-coco-data/yolov3.weights";
    std::string darknetConfig = "yolo-coco-data/yolov3.cfg";
    cv::dnn::Net network = cv::dnn::readNetFromDarknet(darknetConfig, darknetModel);

    // Getting list with names of all layers from YOLO v3 network
    std::cout << std::endl;
    std::vector< cv::String > layers_names_all = network.getLayerNames();
    for (const auto &layers_name : layers_names_all)
    {
       std::cout << layers_name << ", ";
    }
    std::cout << std::endl;

    // Run forward pass to get output
    std::vector< int > outlayers = network.getUnconnectedOutLayers();
    std::vector< cv::String > layers_names_output;

    for(int id : outlayers)
    {
      layers_names_output.push_back(layers_names_all[id - 1]);
    }

    // print(layers_names_output)  == ['yolo_82', 'yolo_94', 'yolo_106']
    std::cout << std::endl;
    for (const auto &layers_name_out : layers_names_output)
    {
       std::cout << layers_name_out << ", ";
    }
    std::cout << std::endl;

    // Set backend and target if needed
    // network.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    // network.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Load image or video
    // cv::Mat frame = cv::imread("images/woman-working-in-the-office.jpg");
    // cv::VideoCapture camera(0);
    cv::VideoCapture camera("videos/traffic-cars-and-people.mp4");
    // cv::VideoCapture camera("videos/traffic-cars.mp4");
 
    // Setting minimum probability to eliminate weak predictions
    // Setting threshold for filtering weak bounding boxes with non-maximum suppression
    float probability_minimum = 0.5;
    float threshold = 0.3;

    while(true)
    {
        cv::Mat frame;
        camera.read( frame );

        // Getting spatial dimensions of the frame
        // we do it only once from the very beginning
        // all other frames have the same dimension
        int w = frame.cols;
        int h = frame.rows;

        // Preprocess image
        cv::Mat blob;
        cv::dnn::blobFromImage( frame, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(0,0,0), true, false );

        // Set input blob
        network.setInput(blob);

        // Forward pass
        std::vector<cv::Mat> output_from_network;
        const auto start{std::chrono::steady_clock::now()};
        network.forward(output_from_network, network.getUnconnectedOutLayersNames());
        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration< double > elapsed_seconds{end - start};
        std::cout << "Elapsed Time : " << elapsed_seconds.count() << " [sec]" << std::endl; 

        // Postprocess output
        std::vector< int > class_numbers;
        std::vector< float > confidences;
        std::vector< cv::Rect > bounding_boxes;

        for (const auto & result : output_from_network) 
        {
            // Loop over each detection
            for (int i = 0; i < result.rows; ++i) 
            {
                // every iteration of i read the next result.cols number of memory block.
                // float* detected_objects = (float*)result.data;
                // i == 0,1, 2, ... & sizeof(float) == 4 & result.cols == 85
                int offset = i * sizeof(float) * (result.cols); 
                float* detected_objects = (float*)(result.data + offset);

                // Extract confidence
                float confidence_current = 0;
                for (int j = 5; j < result.cols; j++)
                {
                    if(detected_objects[j] > confidence_current)
                    {
                       confidence_current = detected_objects[j];
                    }
                }

                if (confidence_current > 0.5) 
                {
                    float x_center = detected_objects[0] * frame.cols;
                    float y_center = detected_objects[1] * frame.rows;
                    float box_width = detected_objects[2] * frame.cols;
                    float box_height = detected_objects[3] * frame.rows;
                    int x_min = int(x_center - (box_width / 2));
                    int y_min = int(y_center - (box_height / 2));
                    int classId = std::max_element(detected_objects + 5, detected_objects + 80) - detected_objects - 5;

                    bounding_boxes.push_back( cv::Rect(x_min, y_min, int(box_width), int(box_height)) );
                    confidences.push_back( confidence_current );
                    class_numbers.push_back(classId);
                    float score = *(std::max_element(detected_objects + 5, detected_objects + 80));
                }
            }
        }

        // Apply non-maximum suppression
        std::vector< int > indices;
        cv::dnn::NMSBoxes( bounding_boxes, confidences, 0.5, 0.4, indices );

        // Draw bounding boxes
        for (int i : indices) 
        {
            cv::rectangle( frame, bounding_boxes[i], cv::Scalar(0, 255, 0), 2 );
            cv::putText( frame, 
                         "Class: " + std::to_string(class_numbers[i]), 
                         cv::Point(bounding_boxes[i].x, bounding_boxes[i].y - 10), 
                         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 
                         2);
        }

        // Show output
        cv::imshow("Output", frame);
        if ( cv::waitKey(1) == 'q')
           break;

    }

    // Releasing camera
    camera.release();
    // Destroying all opened OpenCV windows
    cv::destroyAllWindows();


    return 0;
}