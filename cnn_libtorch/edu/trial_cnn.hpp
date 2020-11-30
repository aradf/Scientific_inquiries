#include <torch/torch.h>

struct CNetwork : torch::nn::Module 
{
  CNetwork(int64_t N, int64_t M) : linear(register_module("linear", torch::nn::Linear(N, M))) 
  {
    another_bias = register_parameter("b", torch::randn(M));
  }
  
  torch::Tensor forward(torch::Tensor input) 
  {
    return linear(input) + another_bias;
  }
  
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

struct Net : torch::nn::Module 
{
  Net(int64_t N, int64_t M) 
  {
    linear = register_module("linear", torch::nn::Linear(N, M));
  }

  torch::nn::Linear linear{nullptr}; // construct an empty holder
};

void a(struct Net net)
{
  /* data */
  std::cout << net << std::endl;
};
