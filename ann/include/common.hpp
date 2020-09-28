#ifndef __COMMON_HPP
#define __COMMON_HPP
#include "data.hpp"
#include <vector>

class CCommon_data 
{
 protected:
    std::vector<CData *> *training_data;
    std::vector<CData *> *test_data;
    std::vector<CData *> *validation_data;
 public:
    void set_training_data(std::vector<CData *> * vect);
    void set_test_data(std::vector<CData *> * vect);
    void set_validation_data(std::vector<CData *> * vect);
};
#endif
