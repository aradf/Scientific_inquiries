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

/*
 *  template definition: 
 *  class T is a blueprint for generic class type
 *  example T: std::vector<int>
 */ 
template<class T>
void print_stlContainer(T t, std::string msg)
{
   std::cout << msg << " { ";
   typename T::iterator temp_iterator;
   for (temp_iterator = t.begin(); temp_iterator != t.end(); ++temp_iterator)
   {
      std::cout << *temp_iterator << ", ";
   }
   std::cout << " } " << std::endl;
}

template<class T>
void remove_anyConatiner(T &t, int k)
{
   std::cout << "Erased item: " << k << std::endl;
   typename T::iterator temp_iterator;
   for (temp_iterator = t.begin(); temp_iterator != t.end();)
   {
      if (*temp_iterator == k)
      {
         temp_iterator = t.erase(temp_iterator);
      }
      else
      {
         temp_iterator++;
      }
   }
}

bool equal_one(int e, int p)
{
   return (e==p);
}

template<class T>
void remove_ifContainer(T &t, int k)
{
   std::cout << "Erased item: " << k << std::endl;
   // typename T::iterator temp_iterator;
   auto temp_iterator = std::remove_if(t.begin(), t.end(), std::bind(equal_one, std::placeholders::_1, k));
   t.erase(temp_iterator, t.end());
}

}  // end: namespace scl

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   std::vector<int> vector = {1, 4, 6, 1, 1, 1, 1, 12, 18, 16};
   //stl template vector containing integers.
   scl::print_stlContainer< std::vector<int> >(vector, "Original: ");

   //stl template vector containing integers.
   scl::remove_anyConatiner< std::vector<int>  >(vector, 1);

   //stl template vector containing integers.
   scl::print_stlContainer< std::vector<int> >(vector, "Removed: ");

   vector = {1, 4, 6, 1, 1, 1, 1, 12, 18, 16};
   //stl template vector containing integers.
   auto itr = std::remove(vector.begin(), vector.end(), 1);
   vector.erase(itr, vector.end());
   vector.shrink_to_fit();    // c++ 11;

   //stl template vector containing integers.
   scl::print_stlContainer< std::vector<int> >(vector, "Efficient: ");
   std::cout << "Capacity: " << vector.capacity() << std::endl;

   std::list<int> my_list = {1, 4, 6, 1, 1, 1, 1, 12, 18, 16};

   //stl template list containing integers.
   scl::remove_anyConatiner< std::list<int>  >(my_list, 1);

   //stl template list containing integers.
   scl::print_stlContainer< std::list<int> >(my_list, "Removed: ");

   my_list.remove(1);
   scl::print_stlContainer< std::list<int> >(my_list, "Efficient: ");

   std::multiset<int> my_multiSet = {1, 4, 6, 1, 1, 1, 1, 12, 18, 16};

   scl::print_stlContainer< std::multiset<int> >(my_multiSet, "Multi-set");
   scl::remove_anyConatiner< std::multiset<int> >(my_multiSet, 1);

   scl::print_stlContainer< std::multiset<int> >(my_multiSet, "Multi-set");

   my_multiSet = {1, 4, 6, 1, 1, 1, 1, 12, 18, 16};
   my_multiSet.erase(1);
   scl::print_stlContainer< std::multiset<int> >(my_multiSet, "Multi-set");

   std::vector<int> vector1 = {1, 4, 6, 1, 1, 1, 1, 12, 18, 16};
   scl::remove_ifContainer< std::vector<int> >(vector1, 12);
   // scl::print_stlContainer< std::multiset<int> >(my_multiSet, "Multi-set");
   scl::print_stlContainer< std::vector<int> >(vector1, "vector1");
   


   std::cout << "Good Bye" << std::endl;
   return 0;
}