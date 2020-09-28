#ifndef __DATA_HPP
#define __DATA_HPP

#include <vector>
#include "stdint.h" // uint8_t 
#include "stdio.h"

class CData
{
private:

  std::vector<uint8_t> *feature_vector;
  std::vector<double> *normalized_feature_vector;
  std::vector<int> *class_vector;

  uint8_t label; 
  uint8_t enumerated_label; // A -> 1
  double distance;

public:
  CData();
  ~CData();

  void set_distance(double);
  void set_feature_vector(std::vector<uint8_t>*);
  void set_normalized_feature_vector(std::vector<double>*);
  void set_class_vector(int counts);
  void append_to_feature_vector(uint8_t);
  void append_to_feature_vector(double);
  void set_label(uint8_t);
  void set_enumerated_label(uint8_t);
  void print_vector();
  void print_normalized_vector();

  double get_distance();
  int get_feature_vector_size();
  uint8_t get_label();
  uint8_t get_enumerated_label();

  std::vector<uint8_t> * get_feature_vector();
  std::vector<double> * get_normalized_feature_vector();
  std::vector<int> get_class_vector();

};

#endif
