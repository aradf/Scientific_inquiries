#include "../include/layer.hpp"
#include "../include/helper_funtion.hpp"

CLayer::CLayer(int previous_layer_size, int current_layer_size)
{
    debug_print(" CLayer::CLayer()");

    for(int i = 0; i < current_layer_size; i++)
    {
        neurons.push_back(new CNeuron(previous_layer_size, current_layer_size));
    }
    this->current_layer_size = current_layer_size;
}

CLayer::CLayer(int previous_layer_size, int current_layer_size, CTraining_handler *training_handler)
{
    debug_print(" CLayer::CLayer()");

    for(int i = 0; i < current_layer_size; i++)
    {
        neurons.push_back(new CNeuron(previous_layer_size, current_layer_size, training_handler));
    }
    this->current_layer_size = current_layer_size;
}

CLayer::~CLayer()
{
    debug_print(" CLayer::~CLayer() ");
    for (int i = 0; i < current_layer_size; i++)
    {
        delete neurons.at(i);
    }
}
