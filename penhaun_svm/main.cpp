#include "penhaun_svm.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace std;

double frand()
{ 
  return (double)rand()/RAND_MAX; 
}

int main()
{
  Frame_libsvm::CSvm_wrapper svm_wrapper;
  svm_wrapper.set_wrapper_nu_svc();
  
  for (int i=0; i<1000; i++)
  {
    double x_value = frand()*2-1;
    double y_value = frand()*2-1;
    vector<pair<int, double> > vect_data;
    vect_data.push_back(make_pair(0, x_value));
    vect_data.push_back(make_pair(1, y_value));
    svm_wrapper.add_wrapper_train_data(x_value * y_value >= 0, vect_data);
  }
  
  svm_wrapper.wrapper_train();

  for (int i=0; i<10; i++)
  {
    double x_value = frand()*2-1;
    double y_value = frand()*2-1;
    vector<pair<int, double> > data_vect;
    data_vect.push_back(make_pair(0, x_value ));
    data_vect.push_back(make_pair(1, y_value ));
    cout<< x_value <<", "<< y_value <<": "<<svm_wrapper.wrapper_predict(data_vect)<<endl;
  }

  return 0;
}
