#include <torch/torch.h>        /*provides defintion for tensorflow math.*/
#include <iostream>             /*provides definitions for the input and output.*/
#include <vector>               /*provides definitions for vector pattern.*/
#include <chrono>  // for high_resolution_clock

#include "FilenameDataset.h"

using namespace torch;
using namespace std;


/******************************************
 * 'torch::nn::Module' is the base class for all neural network modules.
 * AlexNetImpl inherits from the base class.
 * The macro register_module() registers modules from nn like nn::Linear. 
 * conv1, ..., conv5 are the super-class constructor being populted.
 * nn::Conv2d is a 2-Dim Convolutional layer 
 * nn::Conv2dOptions allows specialization for the nn::Conv2d object.
 * nn::Linear Applies a linear transformation to the incoming data
 * 
 * 
 * 
 * 
******************************************/
struct AlexNetImpl : nn::Module {

    AlexNetImpl(int64_t N)
            : conv1(register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)))),
            conv2(register_module("conv2", nn::Conv2d(nn::Conv2dOptions(64, 192, 5).padding(2)))),
            conv3(register_module("conv3", nn::Conv2d(nn::Conv2dOptions(192, 384, 3).padding(1)))),
            conv4(register_module("conv4", nn::Conv2d(nn::Conv2dOptions(384, 256, 3).padding(1)))),
            conv5(register_module("conv5", nn::Conv2d(nn::Conv2dOptions(256, 256, 3).padding(1)))),
            linear1(register_module("linear1", nn::Linear(9216, 4096))),
            linear2(register_module("linear2", nn::Linear(4096, 4096))),
            linear3(register_module("linear3", nn::Linear(4096, 1000))),
            dropout(register_module("dropout", nn::Dropout(nn::DropoutOptions(0.5)))){}

    torch::Tensor forward(const torch::Tensor& input) {
        auto x = torch::relu(conv1(input));
        x = torch::max_pool2d(x, 3, 2);

        x = relu(conv2(x));
        x = max_pool2d(x, 3, 2);

        x = relu(conv3(x));
        x = relu(conv4(x));
        x = relu(conv5(x));
        x = max_pool2d(x, 3, 2);
        // Classifier, 256 * 6 * 6 = 9216
        x = x.view({x.size(0), 9216});
        x = dropout(x);
        x = relu(linear1(x));

        x = dropout(x);
        x = relu(linear2(x));

        x = linear3(x);
        return x;
    }
    torch::nn::Linear linear1, linear2, linear3;
    nn::Dropout dropout;
    nn::Conv2d conv1, conv2, conv3, conv4, conv5;
};

TORCH_MODULE_IMPL(AlexNet, AlexNetImpl);