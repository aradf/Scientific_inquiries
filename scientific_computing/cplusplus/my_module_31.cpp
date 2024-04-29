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

}  // end: namespace scl

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;

   {
      // struct Lsb_less
      class Lsb_less
      {
         public:
         bool operator()(const int& x,const int& y) const
         {
            return (x%10 < y%10);
         }
      };
  
      std::set<int, Lsb_less> s = {21, 23, 26, 27};
      int index = 0;

      for (auto iCnt : s)  std::cout << iCnt << " ";
      std::cout << std::endl;

      std::set<int, Lsb_less>::iterator iter1, iter2;

      iter1 = std::find(s.begin(), s.end(), 36);
      std::cout << *iter1 << std::endl;
      
      iter2 = s.find(36);
      std::cout << *iter2 << std::endl;

      std::cout << std::endl;
   }
 
   {
      std::set<int> s = {21, 23, 26, 27};
      int index = 0;

      for (auto iCnt : s)  std::cout << iCnt << " ";
      std::cout << std::endl;

      std::set<int>::iterator iter1, iter2;

      iter1 = std::find(s.begin(), s.end(), 36);
      std::cout << *iter1 << std::endl;
      
      iter2 = s.find(36);
      std::cout << *iter2 << std::endl;

      std::cout << std::endl;
    }

  {
      // struct some_operator
      class some_operator
      {
         public:
         bool operator()(const int& x,const int& y) const
         {
            return (x < 25 && y < 25);
         }
      };
  
      std::set<int, some_operator> s = {21, 23, 26, 27};
      int index = 0;

      for (auto iCnt : s)  std::cout << iCnt << " ";
      std::cout << std::endl;

      std::set<int, some_operator>::iterator iter1, iter2;

      iter1 = std::find(s.begin(), s.end(), 36);
      std::cout << *iter1 << std::endl;
      
      iter2 = s.find(36);
      std::cout << *iter2 << std::endl;

      std::cout << std::endl;
   }
 

   std::cout << "Good Bye" << std::endl;
   return 0;
}