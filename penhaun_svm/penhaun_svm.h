#pragma once

#include "libsvm/svm.h"

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <cstdlib>
#include <malloc.h>
#include <unistd.h>

namespace Frame_libsvm
{

class CSvm_wrapper
{

private:
  std::vector<std::pair<int, std::vector<std::pair<int, double> > > > wrapper_data;
  svm_parameter wrapper_param;
  svm_model *wrapper_model;

public:
  CSvm_wrapper():wrapper_model(NULL)
  {
    wrapper_param.svm_type=C_SVC;
    wrapper_param.kernel_type=RBF;
    wrapper_param.degree=3;
    wrapper_param.gamma=-1; // 1/num_features
    wrapper_param.coef0=0;
    wrapper_param.cache_size=100;
    wrapper_param.eps=0.001;
    wrapper_param.C=1;
    wrapper_param.nr_weight=0;
    wrapper_param.weight_label=NULL;
    wrapper_param.weight=NULL;
    wrapper_param.nu=0.5;
    wrapper_param.p=0.1;
    wrapper_param.shrinking=1;
    wrapper_param.probability=0;
  }

  ~CSvm_wrapper();
  explicit CSvm_wrapper(const std::string &file_name);
  void set_wrapper_linear();
  void set_wrapper_rbf(double gamma=-1);
  void set_wrapper_sigmoid(double gamma=-1, double coef0=0);
  void set_wrapper_poly(double gamma=-1, double coef0=0, double degree=3);
  void set_wrapper_c_svc(double C=1);
  void set_wrapper_nu_svc(double nu=0.5);
  void set_wrapper_nu_svr(double nu=0.5, double C=1);
  void set_wrapper_epsilon_svr(double eps=0.001, double C=1);
  void set_wrapper_one_class(double nu=0.5);
  void set_wrapper_cache_size(double mb);
  void set_wrapper_shrinking(bool b);
  void set_wrapper_probability(bool b);
  void add_wrapper_train_data(double ans, const std::vector<std::pair<int, double> > &vect);

  void wrapper_train(bool auto_reload=true);

  double wrapper_predict(const std::vector<std::pair<int, double> > &vect) const 
  {
    std::vector<svm_node> node(vect.size()+1);
    
    for (size_t i=0; i<vect.size(); i++)
    {
      node[i].index=vect[i].first;
      node[i].value=vect[i].second;
    }
    node[vect.size()].index=-1;
    node[vect.size()].value=0;

    return svm_predict(wrapper_model, &node[0]);
  }

  void wrapper_save(const std::string &file_name) const 
  {
    svm_save_model(file_name.c_str(), wrapper_model);
  }

  void wrapper_swap(CSvm_wrapper &r) throw() 
  {
    wrapper_data.swap(r.wrapper_data);
    std::swap(wrapper_param, r.wrapper_param);
    std::swap(wrapper_model, r.wrapper_model);
  }

};

} // Frame_libsvm
