#include <numeric>
#include "../include/network.hpp"
#include "../include/layer.hpp"
#include "../include/data_handler.hpp"
#include "../include/helper_funtion.hpp"


CNetwork::CNetwork(std::vector<int> spec, int input_size, int num_classes, double learning_rate)
{
    debug_print("CNetwork::CNetwork() ");
    for(int i = 0; i < spec.size(); i++)
    {
        if(i == 0)
            layers.push_back(new CLayer(input_size, spec.at(i)));
        else
            layers.push_back(new CLayer(layers.at(i-1)->neurons.size(), spec.at(i)));
        
    }
    
    layers.push_back(new CLayer(layers.at(layers.size()-1)->neurons.size(), num_classes));
    this->learning_rate = learning_rate;
}

CNetwork::CNetwork(std::vector<int> spec, int input_size, int num_classes, double learning_rate, CTraining_handler *training_handler)
{
    debug_print("CNetwork::CNetwork() ");
    for(int i = 0; i < spec.size(); i++)
    {
        if(i == 0)
            layers.push_back(new CLayer(input_size, spec.at(i), training_handler));
        else
            layers.push_back(new CLayer(layers.at(i-1)->neurons.size(), spec.at(i), training_handler));
        
    }
    
    layers.push_back(new CLayer(layers.at(layers.size()-1)->neurons.size(), num_classes, training_handler));
    this->learning_rate = learning_rate;

}

CNetwork::~CNetwork() 
{
    debug_print("CNetwork::~CNetwork() ");

    int tmp_layer_size = layers.size();
    for (int i = 0; i < tmp_layer_size ; i++)
    {
        delete layers.at(i);
    }
}

double CNetwork::activate(std::vector<double> weights, std::vector<double> input)
{
    double activation = weights.back(); 
    for(int i = 0; i < weights.size() - 1; i++)
    {
        activation += weights[i] * input[i];
    }
    return activation;
}

double CNetwork::sigmoid(double activation)
{
    return 1.0 / (1.0 + exp(-activation));
}

double CNetwork::transfer_derivative(double output)
{
    return output * (1 - output);
}

void CNetwork::output_train_data(CData_handler *data_handler)
{
    data_handler->open_weight_data_file("data/train_weight.txt");
    std::string str_line;
    std::string number;

    str_line = "#training information \n";
    data_handler->write_weight_data(str_line);

    number = std::to_string(data_handler->get_training_data()->at(0)->get_normalized_feature_vector()->size());
    str_line = "input_layer_size " + number + "\n"; 
    data_handler->write_weight_data(str_line);

    number = std::to_string(layers.at(0)->current_layer_size);
    str_line = "hidden_layer_size " + number + "\n"; 
    data_handler->write_weight_data(str_line);
    
    number = std::to_string(layers.at(1)->current_layer_size);
    str_line = "num_classes " + number + "\n"; 
    data_handler->write_weight_data(str_line);

    number = std::to_string(this->learning_rate);
    str_line = "learning_rate " + number + "\n";
    data_handler->write_weight_data(str_line);

    printf("Rows: Output: Delta: Weights: \n");
    str_line = "Rows: Output: Delta: Weights: ";
    str_line += "\n";
    data_handler->write_weight_data(str_line);

    for(int i = 0; i < layers.size(); i++)
    {
        CLayer *layer = layers.at(i);
        for(CNeuron *neuron_item : layer->neurons)
        {
            neuron_item->print_neuron(data_handler);
        }
    }   
}

std::vector<double> CNetwork::forward_propagation(CData *data)
{
    std::vector<double> inputs = *data->get_normalized_feature_vector();
    for(int i = 0; i < layers.size(); i++)
    {
        CLayer *layer = layers.at(i);
        std::vector<double> new_inputs;
        for(CNeuron *neuron_item : layer->neurons)
        {
            double activation = this->activate(neuron_item->weights, inputs);
            neuron_item->output = this->sigmoid(activation);
            new_inputs.push_back(neuron_item->output);
        }
        inputs = new_inputs;
    }
    return inputs; // output layer outputs
}

void CNetwork::back_propagation(CData *data)
{
    for(int i = layers.size() - 1; i >= 0; i--)
    {
        CLayer *layer = layers.at(i);
        std::vector<double> errors;
        if(i != layers.size() - 1)
        {
            for(int j = 0; j < layer->neurons.size(); j++)
            {
                double error = 0.0;
                for(CNeuron *n : layers.at(i + 1)->neurons)
                {
                    error += (n->weights.at(j) * n->delta);
                }
                errors.push_back(error);
            }
        } else {
            for(int j = 0; j < layer->neurons.size(); j++)
            {
                CNeuron *n = layer->neurons.at(j);
                // expected - actual
                errors.push_back((double)data->get_class_vector().at(j) - n->output); 
            }
        }
        for(int j = 0; j < layer->neurons.size(); j++)
        {
            CNeuron *n = layer->neurons.at(j);
            //gradient / derivative part of back prop.
            n->delta = errors.at(j) * this->transfer_derivative(n->output); 
        }
    }
}

void CNetwork::update_weights(CData *data)
{
    std::vector<double> tmp_inputs = *data->get_normalized_feature_vector();
    for(int i = 0; i < layers.size(); i++)
    {
        if(i != 0)
        {
            for(CNeuron *neutron_item : layers.at(i - 1)->neurons)
            {
                tmp_inputs.push_back(neutron_item->output);
            }
        }
        
        for(CNeuron *neutron_item : layers.at(i)->neurons)
        {
            for(int j = 0; j < tmp_inputs.size(); j++)
            {
                neutron_item->weights.at(j) += this->learning_rate * neutron_item->delta * tmp_inputs.at(j);
            }
            neutron_item->weights.back() += this->learning_rate * neutron_item->delta;
        }
        tmp_inputs.clear();
    }
}

int CNetwork::predict(CData * data)
{
    std::vector<double> outputs = forward_propagation(data);
    int index = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    
    return index;
}

void CNetwork::train(int num_epochs)
{
    for(int i = 0; i < num_epochs; i++)
    {
        double sum_error = 0.0;
        for(CData *data : *this->training_data)
        {
            std::vector<double> outputs = forward_propagation(data);
            std::vector<int> expected = data->get_class_vector();
            double temp_error_sum = 0.0;
            for(int j = 0; j < outputs.size(); j++)
            {
                temp_error_sum += pow((double) expected.at(j) - outputs.at(j), 2);
            }
            sum_error += temp_error_sum;
            back_propagation(data);
            update_weights(data);
        }
        printf("Iteration: %d \t Error=%.4f\n", i, sum_error);
    }
    printf("Done training the Artifical Neural Network \n");

}

double CNetwork::test()
{
    double num_correct = 0.0;
    double tmp_count = 0.0;
    for(CData *data_item : *this->test_data)
    {
        tmp_count++;
        int index = predict(data_item);
        if(data_item->get_class_vector().at(index) == 1) 
            num_correct++;
    }

    test_performance = (num_correct / tmp_count);
    return test_performance;
}

void CNetwork::validate()
{
    double tmp_num_correct = 0.0;
    double tmp_count = 0.0;
    for(CData *data_item : *this->validation_data)
    {
        tmp_count++;
        data_item->print_normalized_vector();
        int index = predict(data_item);
        if(data_item->get_class_vector().at(index) == 1) 
            tmp_num_correct++;
    }
    printf("Validation Performance: %.4f\n", tmp_num_correct / tmp_count);
}

void CNetwork::recognize(CData_handler *example_data_handler)
{
    for(CData *data_item : *example_data_handler->get_data_array())
    {
        data_item->print_normalized_vector();
        int index = predict(data_item);;
        printf("index = %d \n",index);
    }
}
