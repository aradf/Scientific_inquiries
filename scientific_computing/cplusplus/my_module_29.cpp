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
 * g++ -std=c++17 -g -o ./run_compute my_module_28.cpp
 */

// Scientific Computing Library
namespace scl
{
bool lsb_less(int x, int y)
{
   return (x%10 < y%10);
}

bool lessThan10(int x)
{
   return (x<10);
}

}  // end: namespace scl

/*
 * Sorting in STL
 * Sorting algorithm or template functions require random access iterators:
 * vector, deque, container array, native array.
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   {
      std::vector<int> vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};

      std::sort(vec.begin(), vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};

      std::sort(vec.begin(), vec.end(), [](int x, int y){return (x%10 < y%10); });
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};

      std::sort(vec.begin(), vec.end(), [](int x, int y){return scl::lsb_less(x,y); });
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};

      std::sort(vec.begin(), vec.end(), scl::lsb_less );
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};
      std::partial_sort(vec.begin(), vec.begin()+5, vec.end(), std::greater<int>() );
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};
      std::partial_sort(vec.begin(), vec.begin()+5, vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};
      std::nth_element(vec.begin(), vec.begin()+5, vec.end(), std::greater<int>());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};
      std::partition(vec.begin(), vec.end(), scl::lessThan10);
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};
      std::stable_partition(vec.begin(), vec.end(), scl::lessThan10);
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      // Heap Algorithm
      // Heap: 
      // first element is always the largest.
      // add/remvo takes o(log(n)) time.

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};
      std::make_heap(vec.begin(), vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};
      std::pop_heap(vec.begin(), vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};
      vec.pop_back();
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      // add a new element in the heap.
      vec.push_back(100);
      std::push_heap(vec.begin(), vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      // Heap sorting.
      vec = {9, 1, 10, 2, 45, 3, 90, 4, 9, 5, 8};
      std::make_heap(vec.begin(), vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      // stl template algorithm operating on a range.
      // operate sort_heap on every ement in the ange of data
      std::sort_heap(vec.begin(),  vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::cout << std::endl;
   }

   {
      // sorted data algorithm
      // - algorthems that require the data being pre-sorted.
      // - binary search, merge, set operations.
      
      std::vector<int> vec = {8, 9, 9, 9, 45, 87, 90};
      std::vector<int> vec_s = {9, 45, 66};
      std::vector<int>::iterator itr;
      std::pair<std::vector<int>::iterator, std::vector<int>::iterator> pair_of_itr;
      int index = 0;

      // 1. binary search 
      // stl algoritm or function operating on a range of data.
      // 'binary_search' looking for 9 in the range.
      bool found = std::binary_search(vec.begin(), vec.end(), 9);
      
      found = std::includes(vec.begin(), vec.end(),          // Range #1
                            vec_s.begin(), vec_s.end());     // Range #2

      // return true if all elements of vec_s is included in vec
      // both vec_s and vec are sorted.

      itr = std::lower_bound(vec.begin(), vec.end(), 9); 
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;
      // find the first position where 9 could be inserted and still keep the sorting.
      
      itr = std::upper_bound(vec.begin(), vec.end(), 9); 
      std::cout << *itr << std::endl;
      index = itr - vec.begin(); 
      std::cout << index << std::endl;

      // find the last position where 9 could be inserted and still keep the sorting.

      pair_of_itr = std::equal_range(vec.begin(), vec.end(), 9);

      std::cout << std::endl;
   }

   {
      // 2. merge
      std::vector<int> vec = { 7, 8, 9, 10};
      std::vector<int> vec2 = {7, 9, 10};
      std::vector<int> vec_out(10);

      // stl template algorithm or function operating on a range of data.
      // 'merge' the items in data vec and vec2 copied to the vec_out.
      std::merge(vec.begin(), vec.end(),       // range 1 source.
                 vec2.begin(), vec2.end(),     // range 2 source.
                 vec_out.begin());             // destination.

      for (auto iCnt : vec_out) std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {1, 2, 3, 4, 1, 2, 3, 4, 5};
      std::inplace_merge(vec.begin(), vec.begin() + 4, vec.end());

      for (auto iCnt : vec_out) std::cout << iCnt << " ";
      std::cout << std::endl;
   }

   {
      // 3. set operations;
      // - both vec and vec3 must be sorted.
      // - the result data is also sorted.

      std::vector<int> vec = { 7, 8, 9, 10};
      std::vector<int> vec2 = {7, 9, 10};
      std::vector<int> vec_out(10);
      std::cout << std::endl;

      std::set_union(vec.begin(), vec.end(), vec2.begin(), vec2.end(), vec_out.begin());
      for (auto iCnt : vec_out) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::set_intersection(vec.begin(), vec.end(), vec2.begin(), vec2.end(), vec_out.begin());
      for (auto iCnt : vec_out) std::cout << iCnt << " ";
      std::cout << std::endl;
   }

   {
      // Numerical Algorithm <numerica>
      // - Accumulate, inner product, partial sum, adjacent differnce 
      // 1. Accumulate
      
      // stl template vector containing integers.
      std::vector<int> vec = { 7, 8, 9, 10};
      std::vector<int> vec2 = {7, 9, 10};
      std::vector<int> vec_out(10);

      // stl algorithm (numeric) operating on a range of data.
      // 'accumulete' items in range of vec,
      // 10 + vec[0] + vec[1] + ....
      int x = std::accumulate(vec.begin(), vec.end(), 10);
      std::cout << x << " ";
      std::cout << std::endl;

      // stl algorithm (numeric) operating on a range of data.
      // 'accumulate' items in range of vec.
      // 10 * vec[0] * vec[1] + ....
      x = std::accumulate(vec.begin(), vec.end(), 10, std::multiplies<int>());
      std::cout << x << " ";
      std::cout << std::endl;
   }

   {
      // Inner product
      
      // stl template vector containing integers.
      std::vector<int> vec = { 9, 60, 70, 8, 45, 87, 90};
      std::vector<int> vec2 = {7, 9, 10};
      std::vector<int> vec_out(10);

      // stl template algorithm or function operating on a range of data.
      // 'inner_product' on items of vec with init value 10
      // x == 10 + vec[0]*vec[4] + vec[2]*vec[5] + vec[3]*vec[6];
      int x = std::inner_product(vec.begin(), vec.begin()+3, vec.end()-3,10);
      std::cout << x << " ";
      std::cout << std::endl;

      // stl template algorithm or function operating on a range of data.
      // 'inner_product' on items of vec with init value 10
      // x == 10 * (vec[0]+vec[4]) * (vec[2]+vec[5]) * (vec[3]+vec[6])
      x = std::inner_product(vec.begin(), vec.begin()+3, vec.end()-3, 10, std::multiplies<int>(), std::plus<int>());
      std::cout << x << " ";
      std::cout << std::endl;

   }

   {
      // Partial Sum
      
      // stl template vector containing integers.
      std::vector<int> vec = { 9, 60, 70, 8, 45, 87, 90};
      std::vector<int> vec2(10);

      std::partial_sum(vec.begin(), vec.end(), vec2.begin());
      for (auto iCnt : vec2) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::partial_sum(vec.begin(), vec.end(), vec2.begin(), std::multiplies<int>());
      for (auto iCnt : vec2) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::adjacent_difference(vec.begin(), vec.end(), vec2.begin(), std::multiplies<int>());
      for (auto iCnt : vec2) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::adjacent_difference(vec.begin(), vec.end(), vec2.begin(), std::plus<int>());
      for (auto iCnt : vec2) std::cout << iCnt << " ";
      std::cout << std::endl;


   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}