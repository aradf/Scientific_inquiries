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
 * g++ -std=c++17 -g -o ./run_compute my_module_19.cpp
 */

// Scientific Computing Library
namespace scl
{
   bool less_than10(int x)
   {
      return x < 10;
   }

}  // end: namespace scl

/*
 * STL Algoritm Walk through
 * None-Modifying algorithm
 * Example: Count, min and max, compare, linear search, attribute.
 * Algorithm         Data                          Operation
 * std::count        (vec.begin()+2, vec.end()-1,  69);
 */


int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   
   {
      // stl template vector containing integers.
      std::vector<int> vec = {9, 60, 90, 8, 45, 87, 90, 69, 69, 55, 7};
      std::vector<int> vec2 = {9, 60, 70, 8, 45, 86};
      
      // an iterator (pointers) for a stl template vector containing integers.
      std::vector<int>::iterator itr, itr2;
      
      // standard template pair containing stl vector containing integers.
      std::pair<std::vector<int>::iterator, 
                std::vector<int>::iterator> pair_of_itr;

      // c++ 03: some algorithms can be found it trl or boost.

      for( auto iCnt : vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // C++ Lambda Function: is a function without a name intended to invoke
      // a function using a pointer. It takes integer parameter x and returns 
      // a boolean if x less then 10.
      // std::count_if is a template algorithm
      auto num = std::count_if(vec.begin(), vec.end(), [](int x){return x<10;} );

      // 1. counting
      // Algorithm Data Operation
      // std::count and std::count_if is an stl algorithm
      // the template algorithm that applies the Lambda function operation on 
      // the range of iterator from begin to end.
      int n = std::count(vec.begin()+2, vec.end()-1, 69);
      n = std::count_if(vec.begin()+2, vec.end()-1, [](int x){ return x==69;} );
      int m = std::count_if(vec.begin(), vec.end(), [](int x){return x<10;});

      std::cout << n << " " << n << std::endl;

      // 2. Min and Max;
      // std::max_element and std::min_element are stl algorithm
      itr = std::max_element(vec.begin()+2, vec.end());
      itr2 = std::max_element(vec.begin(), vec.end(), [](int x, int y){ return (x%10 < y%10) ;});

      std::cout << *itr << " " << *itr2 << std::endl;

      itr = std::min_element(vec.begin(), vec.end());
      std::cout << *itr << std::endl;

      pair_of_itr = std::minmax_element(vec.begin(), vec.end(), [](int x, int y){ return (x%10 < y%10);});

      // 3. Linear Searching
      // returns the first match.
      // the stl template algoritm 'find' applies equal to 55 operator on 
      // the range of data vec.
      itr = std::find(vec.begin(), vec.end(), 55);
      std::cout << *itr << std::endl;
      int index = itr - vec.begin(); 
      std::cout << index << std::endl;

      // Use the Lambda Function.  the STL template algorihm 'find_if' applies the 
      // operator defined by lambda function on the range of data.
      itr = std::find_if(vec.begin(), vec.end(), [](int x){return x > 80;});
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      itr = std::find_if_not(vec.begin(), vec.end(), [](int x){return x > 80;});
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      itr = std::search_n(vec.begin(), vec.end(), 2, 69);
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      // search sub-range
      // STL template vector containing integers.
      std::vector<int> sub = {45, 87, 90};
    
      itr = std::search(vec.begin(), vec.end(), sub.begin(), sub.end());
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      itr = std::find_end(vec.begin(), vec.end(), sub.begin(), sub.end());
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      // search any of
      // STL template vector containing integers.
      std::vector<int> items = {87, 69};

      // the STL algorithm 'find_first_of' acting/operating on 
      // every element in the data range.
      itr = std::find_first_of(vec.begin(), vec.end(), items.begin(), items.end());
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      // the STL algorithm 'find_first_of' operating the lambda function on  
      // every element in the data range.
      itr = std::find_first_of(vec.begin(), vec.end(), items.begin(), items.end(),[](int x, int y){return x==y*4;});
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      itr = std::adjacent_find(vec.begin(), vec.end());
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      itr = std::adjacent_find(vec.begin(), vec.end(), [](int x, int y){return x == y*4;});
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      // 4. comparing ranges
      if (std::equal(vec.begin(), vec.end(), vec2.begin()))
      {
         std::cout << "vec and vec2 are the same " << std::endl;
      }

      if (std::is_permutation(vec.begin(), vec.end(), vec2.begin()))
      {
         std::cout << "vec and vec2 have the same item in different order " << std::endl;
      }

      pair_of_itr = std::mismatch(vec.begin(), vec.end(), vec2.begin());

      std::lexicographical_compare(vec.begin(), vec.end(), vec2.begin(), vec2.end());

      std::cout << std::endl;
 
      // 5. check attribute
      std::is_sorted(vec.begin(), vec.end());

      itr = std::is_sorted_until(vec.begin(), vec.end());
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      std::is_partitioned(vec.begin(), vec.end(), [](int x){return x> 80;}); 
      std::is_heap(vec.begin(), vec.end());

      itr = std::is_heap_until(vec.begin(), vec.end());
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      std::all_of(vec.begin(), vec.end(), [](int x){return x>80;});
      std::any_of(vec.begin(), vec.end(), [](int x){return x>80;});
      std::none_of(vec.begin(), vec.end(), [](int x){return x>80;});


   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}