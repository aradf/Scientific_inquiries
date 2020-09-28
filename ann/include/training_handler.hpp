#ifndef __TRAINING_HANDLER_HPP
#define __TRAINING_HANDLER_HPP

#include "fstream"
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <math.h>
#include <sstream>

class CTraining_handler
{
int hidden_layer_size;
int input_layer_size;
int num_classes;
int pos_current_weight;
double learning_rate;
std::vector<double> neuron_weights;

public:
  CTraining_handler();
  ~CTraining_handler();

public:
  void read_training_csv(std::string);
  int get_hiden_layer_size();
  int get_input_layer_size();
  int get_num_classes();
  double get_learning_rate();
  double next_weight();

};

#endif
