#include <random>
#include "../include/neuron.hpp"
#include "../include/helper_funtion.hpp"

double generate_random_number(double min, double max)
{
    double random = (double) rand() / RAND_MAX;
    return min + random * (max - min);
}

CNeuron::CNeuron(int previous_layer_size, int current_layer_size)
{
    debug_print("CNeuron::CNeuron()\n");
    initialize_weights(previous_layer_size);
}

CNeuron::CNeuron(int previous_layer_size, int current_layer_size, CTraining_handler *training_handler)
{
    debug_print("CNeuron::CNeuron()\n");
    initialize_weights(previous_layer_size, training_handler);
}

CNeuron::~CNeuron()
{
    debug_print("CNeuron::~CNeuron()\n");
}

void CNeuron::initialize_weights(int previous_layer_size, CTraining_handler *training_handler)
{
    this->output = training_handler->next_weight();
    this->delta  = training_handler->next_weight();
    for(int i = 0; i < previous_layer_size + 1; i++)
    {
        weights.push_back(training_handler->next_weight());
    }
}


void CNeuron::initialize_weights(int previous_layer_size)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    for(int i = 0; i < previous_layer_size + 1; i++)
    {
        weights.push_back(generate_random_number(-1.0, 1.0));
    }
}

void CNeuron::print_neuron(CData_handler *data_handler)
{
    std::string str_line_neuron;
    std::string number;

    // Section to save output.
    printf("Rows: %f ", this->output);
    number = std::to_string(this->output);
    str_line_neuron = "Rows: " + number + " ";
    data_handler->write_weight_data(str_line_neuron);

    // Section to save delta
    printf("%f ", this->delta);
    number = std::to_string(this->delta);
    str_line_neuron = number + " ";
    data_handler->write_weight_data(str_line_neuron);

    // Section to save weights
    //printf("weights: ");
    //str_line_neuron = "weights: ";    
    //data_handler->write_weight_data(str_line_neuron);

    for (int i = 0 ; i < this->weights.size(); i ++)
    {
        printf("%f ",this->weights.at(i));
        number = std::to_string(this->weights.at(i));
        str_line_neuron = number + " ";
        data_handler->write_weight_data(str_line_neuron);        
    }

    printf ("\n");
    str_line_neuron = "\n";
    data_handler->write_weight_data(str_line_neuron);
}