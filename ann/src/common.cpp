#include "../include/common.hpp"

void CCommon_data::set_training_data(std::vector<CData *> * vect)
{
 training_data = vect;
}
void CCommon_data::set_test_data(std::vector<CData *> * vect)
{
 test_data = vect;
}
void CCommon_data::set_validation_data(std::vector<CData *> * vect)
{
  validation_data = vect;
}
