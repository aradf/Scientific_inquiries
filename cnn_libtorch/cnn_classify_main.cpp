#include <iostream>
#include "cnn_classify_network.hpp"
#include <torch/script.h>
#include <torch/nn/functional/activation.h>

void cnn_classify_train();

int main(int argc, const char* argv[]) 
{
  std::cout << "Commencing: Convolutional Neural Network." << std::endl ;

  if (argc != 2) 
  {
    std::cout << "usage: train the model.\n" << std::endl;
    cnn_classify_train();
    return 0;
  }

  /*********************
   * 
   * The next few lines of code has a defect.  Gave up on fixing it.
   * 
  ********************/

  torch::jit::script::Module module;
  try 
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    
  }
  catch (const c10::Error& e) 
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "Model loaded successfully" << std::endl;

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 28, 28}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  std::cout << "The End is Nigh" << std::endl ;
  return 0;
}

void cnn_classify_train()
{
  // Create a new instance of CNetwork.  The shared pointer help ensure 
  // the code is free of memory and resource leaks and are exception-safe
  auto network = std::make_shared<CNetwork>();

  // Create an instance of a data loader for the MNIST dataset.
  // stacks all data tensors into one tensor, and all target 
  // (label) tensors into one tensor
  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("./data/MNIST/raw").map(
          torch::data::transforms::Stack<>()),/*batch_size=*/64);

  // Instantiate an SGD (Stochastic gradient descent) optimization algorithm 
  // to update our network's parameters.  This means that model.base’s 
  // parameters will use the learning rate of 0.01.
  // and a momentum is ignored for all parameters.
  torch::optim::SGD sgd_optimizer(network->parameters(), /*lr=*/0.01);

  for (size_t epoch = 1; epoch <= 30; ++epoch) 
  {
    size_t batch_index = 0;

    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *data_loader) 
    {
      // Reset gradients to zero before starting to do backpropragation 
      // because PyTorch accumulates the gradients on subsequent backward passes
      sgd_optimizer.zero_grad();

      // Execute the model on the input data.
      torch::Tensor prediction = network->forward(batch.data);
      
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
      
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      
      // Update the parameters based on the calculated gradients.
      sgd_optimizer.step();
      
      // Output the loss and checkpoint every 100 batches.
      if (++batch_index % 100 == 0) 
      {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        //torch::save(network, "network.pt");
        torch::save(network, "network.pt");
      }
    }
  }
}