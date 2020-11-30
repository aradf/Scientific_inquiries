#include <torch/torch.h>

/* Sample code for training a FCN on MNIST dataset using PyTorch C++ API */
/* This code uses VGG-16 Layer Network */

struct Network: torch::nn::Module 
{
    // VGG-16 Layer
    // conv1_1 - conv1_2 - pool 1 - conv2_1 - conv2_2 - pool 2 - conv3_1 - conv3_2 - conv3_3 - pool 3 -
    // conv4_1 - conv4_2 - conv4_3 - pool 4 - conv5_1 - conv5_2 - conv5_3 - pool 5 - fc6 - fc7 - fc8
    Network() {
        // Initialize CNN
        conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, { 3,3 }).padding(1)));
        conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).padding(1)));
        // Insert pool layer
        conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, { 3,3 }).padding(1)));
        conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, { 3,3 }).padding(1)));
        // Insert pool layer
        conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, { 3,3 }).padding(1)));
        conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).padding(1)));
        conv3_3 = register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).padding(1)));
        // Insert pool layer
        conv4_1 = register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, { 3,3 }).padding(1)));
        conv4_2 = register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1)));
        conv4_3 = register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1)));
        // Insert pool layer
        conv5_1 = register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1)));
        conv5_2 = register_module("conv5_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1)));
        conv5_3 = register_module("conv5_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1)));
        // Insert pool layer
        fc1 = register_module("fc1", torch::nn::Linear(512*7*7,4096));
        fc2 = register_module("fc2", torch::nn::Linear(4096, 4096));
        fc3 = register_module("fc3", torch::nn::Linear(4096, 1000));
    }

    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x) 
    {
        x = conv1_1->forward(x);
        x = torch::relu(x);
        x = conv1_2->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, { 2,2 }, { 2,2 });
 
        x = conv2_1->forward(x);
        x = torch::relu(x);
        x = conv2_2->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, { 2,2 }, { 2,2 });
 
        x = conv3_1->forward(x);
        x = torch::relu(x);
        x = conv3_2->forward(x);
        x = torch::relu(x);
        x = conv3_3->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, { 2,2 }, { 2,2 });
 
        x = conv4_1->forward(x);
        x = torch::relu(x);
        x = conv4_2->forward(x);
        x = torch::relu(x);
        x = conv4_3->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, { 2,2 }, { 2,2 });
 
        x = conv5_1->forward(x);
        x = torch::relu(x);
        x = conv5_2->forward(x);
        x = torch::relu(x);
        x = conv5_3->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, { 2,2 }, { 2,2 });
 
        x = x.view({ x.size(0), -1 });//512x7x7 = 25088
 
        x = fc1->forward(x);
        x = torch::relu(x);
        x = torch::dropout(x, 0.5, is_training());
 
        x = fc2->forward(x);
        x = torch::relu(x);
        x = torch::dropout(x, 0.5, is_training());
 
        x = fc3->forward(x);
 
        x = torch::log_softmax(x, 1);
 
        return x;
    }

    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::Conv2d conv3_3{nullptr};
    torch::nn::Conv2d conv4_1{nullptr};
    torch::nn::Conv2d conv4_2{nullptr};
    torch::nn::Conv2d conv4_3{nullptr};
    torch::nn::Conv2d conv5_1{nullptr};
    torch::nn::Conv2d conv5_2{nullptr};
    torch::nn::Conv2d conv5_3{nullptr};

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};
