#include <vector>
#include <string>
#include <fstream>


#include "../include/help.hpp"

/**
 * @brief loadyolov5_classList: load list of class list.
 * @param void
 * @return std::vector<std::string>
 */
std::vector<std::string> YOLOV5::loadyolov5_classList()
{
    std::vector<std::string> class_list;

#ifdef BEST_ONNX
    std::ifstream input_file("config_files/classes_best.txt");
#else
    std::ifstream input_file("config_files/classes.txt");
#endif

    std::string line;
    while ( getline(input_file, line) )
    {
        class_list.push_back(line);
    }
    return class_list;
}

/**
 * @brief load_yolvo5Network: load yolov5 model network.
 * @param some_network:
 * @param is_cuda:
 * @return void
 */
void YOLOV5::load_yolvo5Network( cv::dnn::Net& some_network, bool is_cuda )
{
#ifdef BEST_ONNX
    auto result_network = cv::dnn::readNet("config_files/best.onnx");
#else
    auto result_network = cv::dnn::readNet("config_files/yolov5s.onnx");
#endif

    if ( is_cuda )
    {
        std::cout << "Attempty to use CUDA ..." 
                  << std::endl;
        result_network.setPreferableBackend( cv::dnn::DNN_BACKEND_CUDA );
        result_network.setPreferableTarget( cv::dnn::DNN_TARGET_CUDA_FP16 );
    }
    else
    {
        std::cout << "Running on CPU ..." 
                  << std::endl;
        result_network.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
        result_network.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );
    }
    some_network = result_network;
}

/**
 * @brief format_yolov5: change the detected output to yolov5 format.
 * @param source:
 * @return cv::Mat
 */
cv::Mat YOLOV5::format_yolov5( const cv::Mat& source ) 
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result( cv::Rect(0, 0, col, row)) );
    return result;
}

/**
 * @brief inference_yolov5: Runt the yolov5 format to run interferance.
 * @param image:
 * @param net:
 * @param output:
 * @param class_name
 * @return void
 */
void YOLOV5::inference_yolov5( cv::Mat& image, cv::dnn::Net& network, std::vector< Prediction >& output, const std::vector< std::string >& class_name) 
{
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    network.setInput(blob);
    std::vector<cv::Mat> outputs;
    network.forward(outputs, network.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

#ifdef BEST_ONNX
    const int dimensions = 7;
#else
    const int dimensions = 85;
#endif

    const int rows = 25200;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for ( int i = 0; i < rows; ++i ) 
    {
        float confidence = data[4];
        if ( confidence >= CONFIDENCE_THRESHOLD ) 
        {
            float * classes_scores = data + 5;
            cv::Mat scores( 1, 
                            class_name.size(), 
                            CV_32FC1, 
                            classes_scores );

            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

#ifdef BEST_ONNX
        data += 7;
#else
        data += 85;
#endif

    }

    std::vector< int > non_max_suppression_result;
    cv::dnn::NMSBoxes( boxes, 
                       confidences, 
                       SCORE_THRESHOLD, 
                       NMS_THRESHOLD, 
                       non_max_suppression_result);

    for (int i = 0; i < non_max_suppression_result.size(); i++) 
    {
        int idx = non_max_suppression_result[i];
        Prediction result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

