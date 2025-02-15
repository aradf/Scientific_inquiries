
#include "cnn_mnist_network.hpp"

int main() 
{
    std::cout << "Commencing: Convolutional Neural Network." << std::endl ;
	auto net = std::make_shared<Net>();

	// Create multi-threaded data loader for MNIST data
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(torch::data::datasets::MNIST("./data/MNIST/raw").map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
				torch::data::transforms::Stack<>())), 64);

	torch::optim::SGD optimizer(net->parameters(), 0.01); // Learning Rate 0.01

	// net.train();

	for(size_t epoch=1; epoch<=30; ++epoch) {
		size_t batch_index = 0;
		// Iterate data loader to yield batches from the dataset
		for (auto& batch: *data_loader) {
			// Reset gradients
			optimizer.zero_grad();
			// Execute the model
			torch::Tensor prediction = net->forward(batch.data);
			// Compute loss value
			torch::Tensor loss = torch::nll_loss(prediction, batch.target);
			// Compute gradients
			loss.backward();
			// Update the parameters
			optimizer.step();

			// Output the loss and checkpoint every 100 batches
			if (++batch_index % 100 == 0) {
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index 
					<< " | Loss: " << loss.item<float>() << std::endl;
				torch::save(net, "mnist_net.pt");
			}
		}
	}
    std::cout << "The End is Nigh" << std::endl ;
    return 0;
}