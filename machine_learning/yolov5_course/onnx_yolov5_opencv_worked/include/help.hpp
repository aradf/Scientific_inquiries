#ifndef HELP_H
#define HELP_H

#include <opencv2/opencv.hpp>

namespace YOLOV5
{

class vector;
class string;
class fstream;

#define BEST_ONNX

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

/**
 * @brief Prediction: 
 * @param class_id
 * @param confidence;
 * @param Box
 */
struct Prediction
{
    int class_id;
    float confidence;
    cv::Rect box;
};

/**
 * @brief Colors: 
 * @param cv::Scalar
 */
const std::vector< cv::Scalar > colors = { cv::Scalar(255, 255, 0), 
                                           cv::Scalar(0, 255, 0), 
                                           cv::Scalar(0, 255, 255), 
                                           cv::Scalar(255, 0, 0) };

std::vector< std::string > loadyolov5_classList();
void load_yolvo5Network( cv::dnn::Net& some_network, bool is_cuda );
cv::Mat format_yolov5( const cv::Mat& source );
void inference_yolov5( cv::Mat& image, cv::dnn::Net& network, std::vector< Prediction >& output, const std::vector< std::string >& class_name);

}

#endif