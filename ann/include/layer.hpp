#ifndef __LAYER_HPP
#define __LAYER_HPP
#include "neuron.hpp"
#include <stdint.h>
#include <vector>
#include "training_handler.hpp"

class CLayer {

public:
    int current_layer_size;
    std::vector<CNeuron *> neurons;
    std::vector<double> layer_outputs;
    
public:
    ~CLayer();
    CLayer(int, int);
    CLayer(int, int,  CTraining_handler *);
};
#endif
