#ifndef __DATA_HANDLER_HPP
#define __DATA_HANDLER_HPP

//#include "fstream"
#include <fstream>
#include <iostream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <math.h>
#include <cstring>

class CData_handler
{
  std::vector<CData *> *data_array;      // all of the data
  std::vector<CData *> *training_data;
  std::vector<CData *> *test_data;
  std::vector<CData *> *validation_data;
  int classification_counts;
  int feature_vector_size;
  std::map<uint8_t, int> class_from_int;
  std::map<std::string, int> class_from_string; //string key
  std::ofstream *output_weight_file_handler;

  public:
  const double TRAIN_SET_PERCENT = 0.75;
  const double TEST_SET_PERCENT = 0.20;
  const double VALID_SET_PERCENT = 0.05;

  CData_handler();
  ~CData_handler();
  
  void read_csv(std::string, std::string);
  void read_input_data(std::string path);
  void read_label_data(std::string path);
  void open_weight_data_file(std::string path);
  void write_weight_data(std::string weights);
  void split_data();
  void count_classes();
  void normalize();
  void print();
  
  int get_classification_counts();
  int get_data_array_size();
  int get_training_data_size();
  int get_test_data_size();
  int get_validation_size();

  uint32_t format(const unsigned char* bytes);

  std::vector<CData *> * get_data_array();
  std::vector<CData *> * get_training_data();
  std::vector<CData *> * get_test_data();
  std::vector<CData *> * get_validation_data();
  std::map<uint8_t, int> get_class_map();

};

#endif
