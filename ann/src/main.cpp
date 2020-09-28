#include "../include/data_handler.hpp"
#include "../include/network.hpp"
#include "../include/training_handler.hpp"

int recognize_pattern();
int train_neural_network_model();

int main(int argc, char** argv)
{
    printf("You have entered %d arguments. \n", argc);

    if (argc != 1)
    {
        /***********************
         * recognize pattern.
         ***********************/
        recognize_pattern();
    }
    else
    {
        /***********************
         * train model. 
         ***********************/
        train_neural_network_model();
    }
}

int recognize_pattern()
{
    CTraining_handler *training_handler = new CTraining_handler();
    CData_handler *example_data_handler = new CData_handler();
    std::string file_path = "data/example.txt";
    example_data_handler->read_csv(file_path,",");

    training_handler->read_training_csv("data/train_weight.txt");
    std::vector<int> hidden_layers = {training_handler->get_hiden_layer_size()};
    CNetwork *art_neural_network = new CNetwork(
        hidden_layers, 
        training_handler->get_input_layer_size(), 
        training_handler->get_num_classes(),
        training_handler->get_learning_rate(),
        training_handler);
    
    art_neural_network->recognize(example_data_handler);
    
    return 0;
}


int train_neural_network_model()
{
    CData_handler *data_handler = new CData_handler();
#ifdef MNIST
    data_handler->read_input_data("data/train-images-idx3-ubyte");
    data_handler->read_label_data("data/train-labels-idx1-ubyte");
    data_handler->count_classes();
#else
    //data_handler->read_csv("../data/iris_data.txt", ",");
    //std::string file_path = "../data/iris_data.txt";
    std::string file_path = "data/iris_data.txt";
    data_handler->read_csv(file_path,",");

#endif
    data_handler->split_data();
    std::vector<int> hidden_layers = {30};

    // auto lambda = [&]() 
    // {
    //     // Contrive an Artificial Neural Network.
    //     CNetwork *art_neural_network = new CNetwork(
    //         hidden_layers, 
    //         data_handler->get_training_data()->at(0)->get_normalized_feature_vector()->size(), 
    //         data_handler->get_classification_counts(),
    //         0.25);
        
    //     // Process the Artificial Neural Network.
    //     art_neural_network->set_training_data(data_handler->get_training_data());
    //     art_neural_network->set_test_data(data_handler->get_test_data());
    //     art_neural_network->set_validation_data(data_handler->get_validation_data());
    //     art_neural_network->train(30);
    //     art_neural_network->validate();
    //     printf("Test Performance: %.3f\n", art_neural_network->test());
    // };

    // lambda();
    int epochs = 60;
    CNetwork *art_neural_network = new CNetwork(
        hidden_layers, 
        data_handler->get_training_data()->at(0)->get_normalized_feature_vector()->size(), 
        data_handler->get_classification_counts(),
        0.25);
        
    // Process the Artificial Neural Network.
    art_neural_network->set_training_data(data_handler->get_training_data());
    art_neural_network->set_test_data(data_handler->get_test_data());
    art_neural_network->set_validation_data(data_handler->get_validation_data());
    art_neural_network->train(epochs);
    art_neural_network->validate();
    art_neural_network->output_train_data(data_handler);

    printf("Test Performance: %.3f\n", art_neural_network->test());
    delete art_neural_network;
    delete data_handler;

    return 0;
}