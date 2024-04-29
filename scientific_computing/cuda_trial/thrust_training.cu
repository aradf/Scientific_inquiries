
#include <iostream>
#include <cmath>

#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <thrust/device_vector.h>
/*
   nvcc -o ./run_cuda thrust_training.cu

  'lsb release -r' 
        provides Ubuntu 20.04
  'nvidia-smi'
  'nvidia-smi --query-gpu=name --format=csv,noheader'
        NVIDIA GeForce RTX 3060 Laptop GPU
  
  sudo apt install nvidia-cuda-toolkit
  which nvcc
  cuobjdump ./run_cuda

  /user/include
  /user/include/c++/9
  /user/include/cuda-gdb

 */

/*
  create a host vector of 100 elements.  
  which counts up in multiples of 7 starting from 21 using thrustt::sequence
  
*/
int first_practice()
{
   // initialize all ten integers of a device_vector to 1
   const int VECTOR_SIZE = 20;
   thrust::host_vector<int> cpu_firstVector(VECTOR_SIZE, 0);
   thrust::host_vector<int> cpu_secondVector(VECTOR_SIZE, 7);

   // set the elements of H to 0, 1, 2, 3, ...
   thrust::sequence(cpu_firstVector.begin(), 
                    cpu_firstVector.end());

   // thrust::sequence(cpu_firstVector.begin(), 
   //                  cpu_firstVector.end(),
   //                  21,
   //                  7);

   // print cpu_firstVector
   for(int iCnt = 0; iCnt < cpu_firstVector.size(); iCnt++)
      std::cout << "cpu_firstVector[" << iCnt << "] = " << cpu_firstVector[iCnt] << std::endl;

   for(int iCnt = 0; iCnt < cpu_secondVector.size(); iCnt++)
      std::cout << "cpu_secondVector[" << iCnt << "] = " << cpu_secondVector[iCnt] << std::endl;

   thrust::transform(cpu_firstVector.begin()+7, 
                     cpu_firstVector.end(), 
                     cpu_secondVector.begin()+7, 
                     cpu_firstVector.begin()+7, 
                     thrust::multiplies<int>());       

   for(int iCnt = 0; iCnt < cpu_firstVector.size(); iCnt++)
      std::cout << "cpu_firstVector[" << iCnt << "] = " << cpu_firstVector[iCnt] << std::endl;

   return 0;
   
}

/*
  create a host vector of 1000 integers.  Use a binary module to determine which 
  element is even 0 and which element is odd.
  
*/
int second_practice()
{
   // initialize all ten integers of a device_vector to 1
   const int VECTOR_SIZE = 20;
   thrust::host_vector<int> cpu_firstVector(VECTOR_SIZE, 0);
   thrust::host_vector<int> cpu_secondVector(VECTOR_SIZE, 0);
   thrust::host_vector<int> cpu_thirdVector(VECTOR_SIZE, 0);

   // set the elements of H to 0, 1, 2, 3, ...
   thrust::sequence(cpu_firstVector.begin(), 
                    cpu_firstVector.end());

   // fill Z with twos
   thrust::fill(cpu_secondVector.begin(), cpu_secondVector.end(), 2);

   // print cpu_firstVector
   for(int iCnt = 0; iCnt < cpu_firstVector.size(); iCnt++)
      std::cout << "cpu_firstVector[" << iCnt << "] = " << cpu_firstVector[iCnt] << std::endl;

   // print cpu_firstVector
   for(int iCnt = 0; iCnt < cpu_secondVector.size(); iCnt++)
      std::cout << "cpu_secondVector[" << iCnt << "] = " << cpu_secondVector[iCnt] << std::endl;

   // compute Y = X mod 2
   thrust::transform(cpu_firstVector.begin(), 
                     cpu_firstVector.end(), 
                     cpu_secondVector.begin(), 
                     cpu_thirdVector.begin(),
                     thrust::modulus<int>());

   // print cpu_firstVector
   for(int iCnt = 0; iCnt < cpu_thirdVector.size(); iCnt++)
      std::cout << "cpu_thirdVector[" << iCnt << "] = " << cpu_thirdVector[iCnt] << std::endl;


   return 0;
}

float my_rand()
{
   return (rand() %100) - 50;
}

int third_practice()
{
   const int VECTOR_SIZE = 10;
   thrust::host_vector<float> cpu_firstVector(VECTOR_SIZE, 0.0);
   thrust::host_vector<float> cpu_averageVector(VECTOR_SIZE, 0.0);
   thrust::host_vector<float> cpu_meanMinusDataPointVector(VECTOR_SIZE, 0.0);
   thrust::host_vector<float> cpu_squaredVector(VECTOR_SIZE, 0.0);

   // set the elements of H to 0, 1, 2, 3, ...
   // thrust::sequence(cpu_firstVector.begin(), 
   //                  cpu_firstVector.end());

   srand(0);
   thrust::generate(cpu_firstVector.begin(),
                    cpu_firstVector.end(),
                    my_rand);

   for (int iCnt = 0; iCnt < cpu_firstVector.size(); iCnt++)
      std::cout << "cpu_firstVector[" << iCnt<< "] = " << cpu_firstVector[iCnt] << std::endl;

   float sum = thrust::reduce(cpu_firstVector.begin(), 
                              cpu_firstVector.end(), 
                              (int) 0, 
                              thrust::plus<int>());

   float average = sum / (float)VECTOR_SIZE;

   std::cout << "sum: " << sum << " - average: " << average << std::endl;
   thrust::fill(cpu_averageVector.begin(), 
                cpu_averageVector.end(), 
                average);

   thrust::transform(cpu_firstVector.begin(), 
                     cpu_firstVector.end(), 
                     cpu_averageVector.begin(), 
                     cpu_meanMinusDataPointVector.begin(), 
                     thrust::minus<float>());       

   for (int iCnt = 0; iCnt < cpu_meanMinusDataPointVector.size(); iCnt++)
      std::cout << "cpu_meanMinusDataPointVector[" << iCnt<< "] = " << cpu_meanMinusDataPointVector[iCnt] << std::endl;

   thrust::transform(cpu_meanMinusDataPointVector.begin(), 
                     cpu_meanMinusDataPointVector.end(), 
                     cpu_meanMinusDataPointVector.begin(), 
                     cpu_squaredVector.begin(), 
                     thrust::multiplies<float>());       

   for (int iCnt = 0; iCnt < cpu_squaredVector.size(); iCnt++)
      std::cout << "cpu_squaredVector[" << iCnt<< "] = " << cpu_squaredVector[iCnt] << std::endl;

   float sum_squared = thrust::reduce(cpu_squaredVector.begin(), 
                                      cpu_squaredVector.end(), 
                                      (int) 0, 
                                      thrust::plus<int>());
   std::cout << "Standard Deviation: " << sqrt(sum_squared/VECTOR_SIZE) << std::endl;

   return 0;
}

int main()
{
   first_practice();
   std::cout << std::endl;
   std::cout << std::endl;
   std::cout << std::endl;
   second_practice();
   std::cout << std::endl;
   std::cout << std::endl;
   std::cout << std::endl;
   third_practice();
   return 0;
}