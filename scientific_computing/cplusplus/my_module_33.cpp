#include <cstdlib>
#include <iostream>   // std::cin and std::cout

/*
 * STL Headers
 */
#include <vector>
#include <deque>
#include <list>
#include <set>             // Set and multi-set
#include <map>             // map and multi-map
#include <unordered_set>   // unordered set / multi-set
#include <unordered_map>   // unordered map / multi-map
#include <iterator>       
#include <algorithm>
#include <numeric>         // some numeric lgorithm
#include <functional>
#include <array>
#include <math.h>          // pow

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_30.cpp
 */

// Scientific Computing Library
namespace scl
{

class Dog
{

};

void c_fun(const int* arr, int size)
{
   for (int i=0; i<size; i++)
      std::cout << arr[i] << " ";

   std::cout << std::endl;
}

}  // end: namespace scl

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   {
      //stl vector containing integer.
      std::vector<int> my_vector = {2, 3, 4, 5};
      std::cout << my_vector.size() << std::endl;
      std::cout << my_vector.capacity() << std::endl;

      my_vector.push_back(6);
      std::cout << my_vector.size() << std::endl;
      std::cout << my_vector.capacity() << std::endl;

      std::cout << std::endl;
   }
 
   {
      std::vector<scl::Dog> vector_dog(6);
      std::cout << vector_dog.size() << " " << vector_dog.capacity() << std::endl;

      std::vector<scl::Dog> vector_dog2;
      vector_dog2.resize(6);
      std::cout << vector_dog.size() << " " << vector_dog.capacity() << std::endl;

      std::vector<scl::Dog> vector_dog3;
      vector_dog3.reserve(6);
      std::cout << vector_dog.size() << " " << vector_dog.capacity() << std::endl;

      // free memory.
      vector_dog3.shrink_to_fit();

      std::cout << std::endl;
   }

   {
      // Frequency of unpredictable growth
      std::vector<int> vec;
      for (int i=0; i<1025; i++)      
      {
         vec.push_back(i);
      }

      std::cout << vec.size() << " " << vec.capacity() << std::endl;
      std::cout << std::endl;
   }


   {
      // invalidtion of pointers/referances/iterators because of growth.
      std::vector<int> vec = {2, 3, 4, 5};
      int * p = &vec[3];
      std::cout << *p << std::endl;
      vec.push_back(6);               // trigers re-allocation.
      std::cout << *p << std::endl;   // undefined behavior.

      std::deque<int> deq = {2, 3, 4, 5};
      p = &deq[3];
      std::cout << *p << std::endl;
      deq.push_back(6);
      std::cout << *p << std::endl;   // o.k.

      std::cout << std::endl;
   }

   {
      // Vector's unique function: portal to ansi-C.
      std::vector<int> vec = {2, 3, 4, 5};
      scl::c_fun(&vec[0], vec.size());

      // passing data from stl template list to 'ansi-C'
      std::list<int> my_list = {2, 3, 4, 5};
      std::vector<int> vec2(my_list.begin(), my_list.end());

      for (auto iCnt : vec2)  std::cout << iCnt << " ";
      std::cout << std::endl;

      scl::c_fun(&vec2[0], vec2.size());
      scl::c_fun(vec2.data(), vec2.size() );

      std::cout << std::endl;
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}