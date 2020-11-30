#include <torch/torch.h>
#include "trial_cnn.hpp"
//#include "CGenerator.hpp"
#include <iostream>
#include <unistd.h>
char *getcwd(char *buf, size_t size);

int main() 
{
  std::cout << "Commencing: Hello World" << std::endl ;
  // torch::Tensor tensor = torch::eye(3);
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  CNetwork network(4,5);
  int kBatchSize = 10;

  std::cout << network.forward(torch::ones({2, 4})) << std::endl;

  for (const auto& p : network.parameters()) 
  {
    std::cout << p << std::endl;
  }

  const int PATH_MAX_SIZE = 100;
  char cwd[PATH_MAX_SIZE];
  if (getcwd(cwd, sizeof(cwd)) != NULL) 
       std::cout << "Current working dir: " << cwd << std::endl;

  auto dataset = torch::data::datasets::MNIST("./data/MNIST/raw").map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());

    //auto data_loader = torch::data::make_data_loader(std::move(dataset));

  auto data_loader = torch::data::make_data_loader(std::move(dataset),
                     torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

  for (torch::data::Example<>& batch : *data_loader) 
  {
     std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
     for (int64_t i = 0; i < batch.data.size(0); ++i) 
     {
        std::cout << batch.target[i].item<int64_t>() << " ";
     }
     std::cout << std::endl;
  }

  std::cout << "The End is Nigh" << std::endl ;
  return 0;
}