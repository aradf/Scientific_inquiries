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

/*
 * stl container's member functions vs algorithms
 * list
 * void remove(const T);  template<class Comp> void remvoe_if(Comp);
 * void unique();         template<class Comp> void unique(Comp);
 * void sort();           template<class Comp> void sort(Comp);
 * void merge(lists&);    template<class Comp> void merge(Comp);
 * reverse();
 */

/*
 * associatvie containers:
 * size_type count(const T&) const:
 * iterator find(const T&) const;
 * iterator lower_bound(const T&) const;
 * iterator upper_bound(const T&) const;
 * pair<iterator, iterator> equal_range(const T&) const;
 */

// Note: they do not have geralized form, becaue comparison is defined by the container. 

/*
 * Unordered Container;
 * size_type count(const T&) const;
 * iterator find(const T&);
 * std::pair<iterator, iterator> equal_range(const T&);
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;

   {   
      // stl template unordered_set containing integers:
      std::unordered_set<int> s = {2, 4, 1, 8, 5, 9};   // hash table.
      // an iterator for stl template unordered_set containing integers.
      std::unordered_set<int>::iterator itr;
      int index = 9;

      for (auto iCnt : s) std::cout << iCnt << " ";
      std::cout << std::endl;

      // Using member function of 'unordered_set' takes constant time O(1)
      itr = s.find(4);
      std::cout << *itr << std::endl;

      // using stl algorithm takes linear time O(n)
      itr = std::find(s.begin(), s.end(), 4);
      std::cout << *itr << std::endl;

   }

   {
      // how about map/multi-map
      // stl template map containing a char and stl string
      std::map<char, std::string> my_map = {{'S',"Sunday"},{'M', "Monday"}, {'W', "Wednesday"}};
      std::map<char, std::string>::iterator itr;


      // using member function O(log(n))
      itr = my_map.find('F');


      // using stl Algorithm O(n)
      // itr = std::find(my_map.begin(), my_map.end(), std::make_pair('F', "Friday"));
   }

   {
      // how about list
      // stl template map containing a char and stl string
      std::list<int> s = {2, 4, 1, 8, 5, 9};
      std::list<int>::iterator itr;
      for (auto iCnt : s)   std::cout << iCnt << " ";
      std::cout << std::endl;

      // using stl's template list 'remove' member function. O(n)
      // s = {2, 1, 8, 4, 5, 9};
      s.remove(4);
      for (auto iCnt : s)   std::cout << iCnt << " ";
      std::cout << std::endl;

      // using stl algoritm  O(n)
      // s = {2, 1, 8, 5, 9, 9};
      s = {2, 4, 1, 8, 5, 9};
      itr = std::remove(s.begin(), s.end(), 4);
      for (auto iCnt : s)   std::cout << iCnt << " ";
      std::cout << std::endl;

      // s = {2, 1, 8, 5, 9};
      s.erase(itr, s.end());
      for (auto iCnt : s)   std::cout << iCnt << " ";      
      std::cout << std::endl;

      std::cout << std::endl;
   }

   {
      // sort
      std::list<int> s = {2, 4, 1, 8, 5, 9};
      std::list<int>::iterator itr;
      for (auto iCnt : s)   std::cout << iCnt << " ";
      std::cout << std::endl;

      s.sort();
      for (auto iCnt : s)   std::cout << iCnt << " ";
      std::cout << std::endl;

      s = {2, 4, 1, 8, 5, 9};
      // std::sort(s.begin(), s.end());      // undefined behavior.
      // for (auto iCnt : s)   std::cout << iCnt << " ";
      // std::cout << std::endl;

   }

   {
      // Reverse Iterators and Iterators.
      typedef std::vector<int>::iterator reverse_itratorType;
      std::reverse_iterator<reverse_itratorType> reverse_iterator;
      std::vector<int>::reverse_iterator reverse_iterator2;

      std::vector<int> vec = {4, 5, 6, 7};
      for (reverse_iterator2 = vec.rbegin(); reverse_iterator2 != vec.rend(); reverse_iterator2++)
      {
         std::cout << *reverse_iterator2 << " ";
      }
      std::cout << std::endl;

   }

   {
      // convert an itr to ritr
      std::vector<int>::iterator iter;
      std::vector<int>::reverse_iterator reverse_iter;

      reverse_iter = std::vector<int>::reverse_iterator(iter);
      iter = reverse_iter.base();
   }


   {
      std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7};
      std::vector<int>::reverse_iterator reverse_iter;
      std::vector<int>::iterator iter;
      int index = 0;
      for (auto iCnt : vec)   std::cout << iCnt << " ";
      std::cout << std::endl;      
      
      reverse_iter = std::find(vec.rbegin(), vec.rend(), 3);

      index = reverse_iter - vec.rbegin();
      std::cout << *reverse_iter << " " << index << std::endl;

      iter = reverse_iter.base();

      index = iter - vec.begin();
      std::cout << *iter << " " << index << std::endl;

      vec.insert(reverse_iter.base(), 9);
      for (auto iCnt : vec)   std::cout << iCnt << " ";
      std::cout << std::endl;      

      vec = {1, 2, 3, 4, 5, 6, 7};
      for (auto iCnt : vec)   std::cout << iCnt << " ";
      std::cout << std::endl;      
      reverse_iter = std::find(vec.rbegin(), vec.rend(), 3);

      vec.erase(reverse_iter.base() -1 );
      // vec.erase(reverse_iter.base());
      for (auto iCnt : vec)   std::cout << iCnt << " ";
      std::cout << std::endl;      

      std::cout << "Good Bye" << std::endl;
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}