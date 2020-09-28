#include "../include/data.hpp"
#include "../include/helper_funtion.hpp"

 CData::CData()
 {
   debug_print("CData::CData()");
   /**************************
   feature_vector = new std::vector<uint8_t>();
   normalized_feature_vector = new std::vector<double>();
   class_vector = new std::vector<int>();
   ***************************/
 }
 
 CData::~CData()
 {
   debug_print("CData::~CData()");
   /**********************************
   if (feature_vector != nullptr)
   {
      delete feature_vector;
      feature_vector = nullptr;
   }

   if (normalized_feature_vector != nullptr)
   {
      delete normalized_feature_vector;
      normalized_feature_vector = nullptr;
   }

   if (class_vector != nullptr)
   {
      delete class_vector;
      class_vector = nullptr;
   }
   ********************************/

 }

void CData::set_distance(double dist)
{
  distance = dist;
}
void CData::set_feature_vector(std::vector<uint8_t>* vect)
{
  feature_vector = vect;
}

void CData::set_normalized_feature_vector(std::vector<double>* vect)
{
  normalized_feature_vector = vect;
}
void CData::append_to_feature_vector(uint8_t val)
{
  feature_vector->push_back(val);
}
void CData::append_to_feature_vector(double val)
{
  normalized_feature_vector->push_back(val);
}
void CData::set_label(uint8_t val)
{
  label = val;
}
void CData::set_enumerated_label(uint8_t val)
{
  enumerated_label = val;
}

void CData::set_class_vector(int classCounts)
{
  class_vector = new std::vector<int>();
  for(int i = 0; i < classCounts; i++)
  {
    if(i == label)
      class_vector->push_back(1);
    else
      class_vector->push_back(0);
  }
}

void CData::print_vector()
{
  printf("[ ");
  for(auto val : *feature_vector)
  {
    printf("%u ", val);
  }
  printf("]\n");
}

void CData::print_normalized_vector()
{
  printf("[ ");
  for(auto val : *normalized_feature_vector)
  {
    printf("%.2f ", val);
  }
  printf("]\n");
  
}

double CData::get_distance()
{
  return distance;
}

int CData::get_feature_vector_size()
{
  return feature_vector->size();
}
uint8_t CData::get_label()
{
  return label;
}
uint8_t CData::get_enumerated_label()
{
  return enumerated_label;
}

std::vector<uint8_t> * CData::get_feature_vector()
{
  return feature_vector;
}
std::vector<double> * CData::get_normalized_feature_vector()
{
  return normalized_feature_vector;
}

std::vector<int>  CData::get_class_vector()
{
  return *class_vector;
}