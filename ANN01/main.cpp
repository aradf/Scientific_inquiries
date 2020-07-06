#include <iostream>
#include "Cmnist_loader.h"
#include "CNetwork.h"
#include <vector>

int main()
{
   std::cout << "hello World" << std::endl;
   CMnistLoader mnistObj;
   mnistObj.load_data();   

   std::vector<int> aSize = {784, 30, 10};
   //std::vector<int> aSize = {784, 50, 30, 10};
   CNetwork networkObj(aSize);
   
   networkObj.StochasticGradientDescent(mnistObj.returnTraingDataPtr(), \
               30, 10, 3.0, mnistObj.returnTestDataPtr());


   return 0;
}
