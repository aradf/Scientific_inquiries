#include <iostream>
#include "CAlexnet.hpp"

int main() 
{
  std::cout << "Commencing: Wake up Neural Network" << std::endl ;

  torch::Device device = torch::kCPU;
  std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  }

  int batch_size = 128;
  int iterations = 50;
  auto model = AlexNet(224);
  torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(1e-3));

  model->train();
  model->to(device);

  torch::Tensor x, target, y, loss;
  target = torch::randn({batch_size, 1000}, device);
  x = torch::ones({batch_size, 3, 224, 224}, device);
  for (int i = 0; i < iterations; ++i) {
      optim.zero_grad();
      y = model->forward(x);
      loss = torch::mse_loss(y, target);
      loss.backward();
      optim.step();
      if(i%10 == 0)
        cout << loss << endl;
  }

  std::cout << "The End is Nigh" << std::endl ;
  return 0;
}

