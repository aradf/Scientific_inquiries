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

}  // end: namespace scl

/*
 *
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   {
      /*
      * Algorithm Walk through:
      * Value-changing algorith - chagnes the element values.
      * copy, move, transform, swap, fill, replace, remove.
      */
      
      // stl template vector containing integers
      std::vector<int> vec = {9, 60, 70, 8, 45, 87, 90};
      std::vector<int> vec2 = {0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0};

      // stl template vector containing integers iterator
      std::vector<int>::iterator itr, it2;

      // stl template pair: template paramete: 
      //                    stl template vector containing integers's iterator
      //                    stl template vector containing integers's iterator
      std::pair<std::vector<int>::iterator, std::vector<int>::iterator> pair_of_iter;

      // 1. copy: stl template algorithm operating on a range of data.
      // copy from source to destination.
      std::copy(vec.begin(), vec.end(), vec2.begin());

      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      for(auto iCnt: vec2)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec2 = {0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0};
      // stl template algorithm operating lambda function 'copy if' from source to destination.
      std::copy_if(vec.begin(), vec.end(), vec2.begin(),[](int x){return x>80;});

      for(auto iCnt: vec2)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // stl template algorithm opearting on a range n times.
      std::copy_n(vec.begin(), 4, vec2.begin());

      for(auto iCnt: vec2)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec2 = {0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0};

      // stl template algorithm operating on range of data.
      // 'copy_backward' onto the destination.
      std::copy_backward(vec.begin(), vec.end(), vec2.end());
      for(auto iCnt: vec2)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::cout << std::endl;

   }

   {
      /*
       * 2. move.
       */
      // stl template vector containing strings, initialized.
      std::vector<std::string> vec = {"apple", "orange", "pear", "grape"};
      std::vector<std::string> vec2 = {"", "", "", "", "", ""};

      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // stl template algoritm opearting on range of data.
      // move from source to destination.
      std::move(vec.begin(), vec.end(), vec2.begin());

      for(auto iCnt: vec2)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec2 = {"", "", "", "", "", ""};
      std::move_backward(vec.begin(), vec.end(), vec2.end());
      for(auto iCnt: vec2)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::cout << std::endl;
   }

   {
      /*
       * 3. transform.
       */
      std::vector<int> vec = {9, 60, 70, 8, 45, 87, 90};
      std::vector<int> vec2 = {9, 60, 70, 8, 45, 87, 90};
      std::vector<int> vec3 = {0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0};
      
      // stl template function transform:
      // apply the Lembda function on range of source data to destination data.
      std::transform(vec.begin(), vec.end(), vec3.begin(), [](int x){ return x-1; });

      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      for(auto iCnt: vec3)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec3 = {0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0};

      // stl template function transform:
      // apply the Lembda function on range of source data to destination data.
      std::transform(vec.begin(), vec.end(), vec2.begin(), vec3.begin(), [](int x, int y){ return x+y; });

      for(auto iCnt: vec3)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::cout << std::endl;
   }

   {
      /*
       * 4. fill
       */
      std::vector<int> vec = {0, 0, 0, 0, 0};
 
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::fill(vec.begin(), vec.end(), 9);
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {0, 0, 0, 0, 0};
      std::fill_n(vec.begin(), 3, 9);
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {0, 0, 0, 0, 0};
      std::generate(vec.begin(), vec.end(), std::rand);
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {0, 0, 0, 0, 0};
      std::generate_n(vec.begin(), 3, std::rand);
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::cout << std::endl;
   }

   {
      /*
       * 5. Replace
       */
      std::vector<int> vec = {0, 0, 0, 0, 0};
      std::vector<int> vec2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::fill(vec.begin(), vec.end(), 6);
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // stl template algoritm 'replace' operating on a range of data.
      // relaceing 6 with 9.
      std::replace(vec.begin(), vec.end(), 6, 9);
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 60, 70, 8, 45, 87, 90};
      // stl template algoritm 'replace_if' operating on a range of data.
      // relaceing if source is grater than 80 with 9.
      std::replace_if(vec.begin(), vec.end(), [](int x){return x>70;}, 9);
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 60, 70, 8, 6, 87, 90};
      // stl template algoritm 'replace_if' operating on a range of data.
      // relaceing if source is grater than 80 with 9.
      std::replace_copy(vec.begin(), vec.end(), vec2.begin(), 6, 9);
      for(auto iCnt: vec2)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::cout << std::endl;
   }

   {
      /*
       * 5. Remove
       */
      std::vector<int> vec = {1, 2, 3, 4, 5};
      std::vector<int> vec2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      //stl template algorthem operting on a range of data.
      //remove any 3 from the elements in the range.
      std::remove(vec.begin(), vec.end(), 3);
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 60, 70, 8, 45, 87, 90};
      //stl template algorthem operting on a range of data.
      //remove_if operator is a Lymbda function element is greater than 80.
      std::remove_if(vec.begin(), vec.end(), [](int x){ return x>80;});
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 60, 70, 8, 45, 87, 90};
      //stl template algorthem operting on a range of data.
      //remove_copy if element is 6 from the source onto the destination.
      std::remove_copy(vec.begin(), vec.end(), vec2.begin(), 60);
      for(auto iCnt: vec)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      for(auto iCnt: vec2)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      vec = {9, 60, 60, 70, 45, 87, 90};
      // stl algorithm operating on range of data
      // 'unique' removing duplicates.
      std::unique(vec.begin(), vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::unique(vec.begin(), vec.end(), std::less<int>());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::unique_copy(vec.begin(), vec.end(), vec2.begin());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::cout << std::endl;
   }

   {
      /*
       * 6. Order-Changing Algorihms:
       * - reverse, rotate, permute, shuffle
       * They change the order of elements in the stl container, not the element.
       */
      std::vector<int> vec = {9, 60, 70, 8, 45, 87, 90};
      std::vector<int> vec2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      // 1. reverse
      std::reverse(vec.begin()+1, vec.end()-1);
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::reverse_copy(vec.begin()+1, vec.end()-1, vec2.begin());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::rotate(vec.begin(), vec.begin()+3, vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::rotate_copy(vec.begin(), vec.begin()+3, vec.end(), vec2.begin());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;


      std::cout << std::endl;
   }

   {
      /*
       * 6. Permute
       * - reverse, rotate, permute, shuffle
       * They change the order of elements in the stl container, not the element.
       */
      std::vector<int> vec = {9, 60, 70, 8, 45, 87, 90};
      std::vector<int> vec2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      std::next_permutation(vec.begin(), vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      std::prev_permutation(vec.begin(), vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;


      std::cout << std::endl;
   }

   {
      /*
       * 6. shuffle
       * - Rearragne the elements randomly
       */
      std::vector<int> vec = {9, 60, 70, 8, 45, 87, 90};
      std::vector<int> vec2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      std::random_shuffle(vec.begin(), vec.end());
      for(auto iCnt: vec) std::cout << iCnt << " ";
      std::cout << std::endl;

      // std::random_shuffle(vec.begin(), vec.end(), std::rand);
      // for(auto iCnt: vec) std::cout << iCnt << " ";
      // std::cout << std::endl;

      std::cout << std::endl;
   }


   std::cout << "Good Bye" << std::endl;
   return 0;
}