#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <bits/stdc++.h> 
#include "data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "common.hpp"
#include "data_handler.hpp"
#include "training_handler.hpp"

class CNetwork : public CCommon_data
{
public:
    std::vector<CLayer *> layers;
    double learning_rate;
    double test_performance;

public:
    CNetwork(std::vector<int> spec, int, int, double);
    CNetwork(std::vector<int> spec, int, int, double, CTraining_handler *);
    ~CNetwork();
    std::vector<double> forward_propagation(CData *data);
    void back_propagation(CData *data);
    double activate(std::vector<double>, std::vector<double>);
    double sigmoid(double);
    double transfer_derivative(double); 
    void update_weights(CData *data);
    int predict(CData *data); 
    void train(int); 
    double test();
    void validate();
    void output_train_data(CData_handler *data_handler);
    void recognize(CData_handler *example_data_handler);
};

#endif
