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

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_19.cpp
 */

// Scientific Computing Library
namespace scl
{
bool is_odd(int iCnt)
{
   return iCnt%2;
}

}  // end: namespace scl

/*
 * Iterators:
 * 1. Random Access Iterator: vector deque, array
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   
   {
      std::vector<int> my_vector = {1, 2, 3, 4, 5, 6, 7};

      for ( auto iCnt : my_vector)    
         std::cout << iCnt << " ";

      std::cout << std::endl;

      std::vector<int>::iterator vector_iterator;
      
      vector_iterator = my_vector.begin();
      std::cout << *vector_iterator << std::endl;
      
      vector_iterator = vector_iterator + 5;
      std::cout << *vector_iterator << std::endl;
      
      vector_iterator = vector_iterator - 1;
      std::cout << *vector_iterator << std::endl;

      std::vector<int>::iterator itr1, itr2;
      itr1 = my_vector.begin() + 1;
      std::cout << *itr1 << std::endl;
      itr2 = my_vector.begin() + 2;
      std::cout << *itr2 << std::endl;

      if (itr1 > itr2)
         std::cout << "Do something " << std::endl;

      ++vector_iterator;
      --vector_iterator;

      // 2. Bidirectional Iterators: list, set/multi-set, map/multimap
      std::list<int> my_list = {5, 2, 9};  
      std::list<int>::iterator list_iterator;

      for ( auto iCnt : my_list)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      list_iterator = my_list.begin(); 
      ++list_iterator;
      --list_iterator;

      // 3. Forward Iterator:; forward_list
      // std::forward_list<int> forward_listIterator;  // can only be incremented not decrementd.
      // Unordered containers provide "at least" forward iterators;

      for ( auto iCnt : my_vector)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // 4. Input Iterator:  read and process values while iterating forward.
      int my_x = *itr1;
      
      // 5. Oputptu Iterator; output values while iterating forward.
      *itr1 = 100;

      for ( auto iCnt : my_vector)    
         std::cout << iCnt << " ";
      std::cout << std::endl;
   }

   {
      // every container has an iterator and const_iterator
      std::set<int> my_set = {2, 4, 5, 1, 9};
      std::set<int>::iterator iter;
      std::set<int>::const_iterator const_iter;  // Ready only access to the container element.

      for (const_iter = my_set.begin();
         const_iter != my_set.end();
         const_iter++)
         {
            std::cout << *const_iter << " ";
         }
      std::cout <<  std::endl;
      const_iter = my_set.begin();      
      std::cout << *const_iter << std::endl;  
      std::advance(const_iter, 3);
      std::cout << *const_iter << std::endl;

      std::set<int>::iterator itr1, itr2;
      itr1 = my_set.find(4);
      itr2 = my_set.find(9);
      std::cout << std::distance(itr1, itr2) << std::endl;

      //only in c++ 11
      // std::for_each(my_set.cbegin(), my_set.cend(), MyFunction);
   }

   {
      // Insert Iterator
      std::vector<int> vector1 = {4, 5};
      std::vector<int> vector2 = {12, 14, 16, 18};

      for ( auto iCnt : vector1)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      for ( auto iCnt : vector2)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::vector<int>::iterator vector_iter;
      vector_iter = std::find(vector2.begin(), vector2.end(), 16);
      std::cout << *vector_iter << std::endl;
      
      std::insert_iterator< std::vector<int> > my_insertIterator(vector2, vector_iter);

      for ( auto iCnt : vector2)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // source, destinatin, vector2 = {12, 14, 4, 5, 16, 18}
      std::copy(vector1.begin(), vector1.end(), my_insertIterator);
      // other insert iterators: back_insert_iterator, front_insert_iterator.

      for ( auto iCnt : vector1)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      for ( auto iCnt : vector2)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // 2. stream Iterator
      std::vector<std::string> vector4 = {"hello world", "what is up", "end is here"};
      // std::copy(std::istream_iterator<std::string>(std::cin), std::istream_iterator<std::string>(), std::back_inserter(vector4));

      // // make it terse;
      // std::unique_copy(std::istream_iterator<std::string>(std::cin), 
      //                  std::istream_iterator<std::string>(),
      //                  std::ostream_iterator<std::string>(std::cout, " ") );

      std::copy(vector4.begin(), vector4.end(), std::ostream_iterator<std::string>(std::cout, " "));
      std::cout << std::endl;
   } 

   {
      // 3. Reverse Iterator
      std::vector<int> vector5 = {4, 5, 6, 7};
      typedef std::vector<int>::iterator iter_type;
      std::reverse_iterator<iter_type> reverse_iter;
      for (reverse_iter = vector5.rbegin(); 
           reverse_iter != vector5.rend(); 
           reverse_iter++)
         std::cout << *reverse_iter << " ";
         
         std::cout << std::endl;
   }
  
   {
      /*
       * Algorithms
       * - mostly loops
       */
      // vector object containing integers.
      std::vector<int> my_vector = {4, 2, 5, 1, 3, 9};

      // declare an itrator object pointing to the min value in vector.
      std::vector<int>::iterator minimum_iterator;
      minimum_iterator = std::min_element(my_vector.begin(), my_vector.end());

      std::cout << *minimum_iterator << std::endl;

      // sort the vector from begining to position of iterator.
      std::sort(my_vector.begin(), minimum_iterator);

      // loop using 'auto' to autmatically recognize object type of my_vector
      for ( auto iCnt : my_vector)    
         std::cout << iCnt << " ";
      std::cout << std::endl;
     
      // reverse from position of the iterator to the end.
      std::reverse(minimum_iterator, my_vector.end());

      // declare a vector object containing integers with size 3.
      std::vector<int> vector2(3);

      // copy from minimum_iterator to the end of my_vector to vector2
      std::copy(minimum_iterator,       //source 
                my_vector.end(),        //source 
                vector2.begin());       // distination.

      for ( auto iCnt : vector2)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // declar a vector object containing integer 
      std::vector<int> vector3;

      //insert elements from the end of my_vector to vector3.
      vector3.insert(vector3.end(), minimum_iterator, my_vector.end());

      for ( auto iCnt : vector3)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

   }

   {
      // Note 4: Algorithm with function.
      // declare a vector containing integers and initialized.
      std::vector<int> my_vector = {2, 4, 5, 9, 2};

      // delcare an iterator object form vector containing integers
      std::vector<int>::iterator my_itr;

      // apply 'is_odd' function to my-vector
      my_itr = std::find_if(my_vector.begin(), my_vector.end(), scl::is_odd);
      std::cout << *my_itr << std::endl;            // my_itr -> 5;

      std::cout << std::endl;
   }

   {
      // Note 5: Algorithm with native c++ array.
      // declary an array of integers and initialize
      int my_array[4] = {6, 3, 7, 4};

      // sort the array.
      std::sort(my_array, my_array+4);

      std::cout << std::endl;
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}