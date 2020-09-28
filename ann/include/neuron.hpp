#ifndef _NEURON_HPP_
#define _NEURON_HPP_
#include <stdio.h>
#include <vector>
#include <cmath>
#include "../include/data_handler.hpp"
#include "training_handler.hpp"

class CNeuron {
  public:
    double output;
    double delta;
    std::vector<double> weights;
    CNeuron(int, int);
    CNeuron(int, int, CTraining_handler *);
    ~CNeuron();
    
    void initialize_weights(int);
    void initialize_weights(int, CTraining_handler *);
    void print_neuron(CData_handler *data_handler);
};

#endif
