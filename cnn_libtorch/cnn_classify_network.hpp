#include <torch/torch.h>

/*
https://pytorch.org/cppdocs/frontend.html
*/

// Define a sub-class from the base class for all neural network modules.
struct CNetwork : torch::nn::Module 
{
  CNetwork() 
  {
    // Construct and register three Linear submodules which are all
    // the layers to be utilized.  A linear submodule perfomrs liner 
    // transformation to the incoming data.  28 * 28 = 784 
    // & 64 & 32 are the size of hidden layers.
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  // Implement the CNetwork's algorithm.  This forward function defines
  // how the model is going to be run, from input to output.
  torch::Tensor forward(torch::Tensor x) 
  {
    // Use one of many tensor manipulation functions.
    // Applies the relu (rectified linear unit) function element-wise
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    // dropout: During training, randomly zeroes some of the elements 
    // of the input tensor with probability p using samples from a 
    // Bernoulli distribution. Each channel will be zeroed out 
    //independently on every forward call.
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::Linear fc3{nullptr};
};
