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


}  // end: namespace scl

/*
 * Containers:
 * Sequence containers (array and linked list)
 * - vector, deque, list, forward list, array.
 * Associative Containers (binary tree)
 * - set, muliset,
 * - map, multimap
 * Unordered Containers (hash table)
 * - Unordered set/multiset
 * - Unordered map/multimap
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   
   {
      /*
       *
       */
      // vector.size() == 0, it is an one dim array owning integers.
      std::vector<int> my_vector;         
      my_vector.push_back(4);
      my_vector.push_back(1);
      my_vector.push_back(8);

      // Vector specific opertions:
      std::cout << my_vector[2] << std::endl;
      std::cout << my_vector.at(2) << std::endl;

      for (int iCnt = 0; iCnt < my_vector.size(); iCnt++)
      {
         std::cout << my_vector[iCnt] << " ";
      }
      std::cout << std::endl;

      for (std::vector<int>::iterator itr = my_vector.begin();
           itr != my_vector.end();
           ++itr )
      {
         std::cout << *itr << " ";
      }
      std::cout << std::endl;

      // // c++ 11
      // for(my_iter my_vector)
      //    std::cout << my_iter << " ";
      
      // std::cout << std::endl;         

      // vector is a dynamically allocated contiguous array in memory
      int * ptr_vector = &my_vector[0];
      ptr_vector[2] = 6;

      for(std::vector<int>::iterator itr = my_vector.begin();
          itr != my_vector.end();
          ++itr)
      {
         std::cout << *itr << " ";
      }

      if (my_vector.empty()) std::cout << "Not Possible" << std::endl;

      std::cout << my_vector.size() << std::endl;
      // standard template library's vector container owns integers.
      std::vector<int> vector2(my_vector);

      my_vector.clear();
      std::cout << my_vector.size() << std::endl;

      // vector2 becomes empty and my_vector has three elements.
      vector2.swap(my_vector);

      std::cout << std::endl;
   }

   {
      /*
       *
       */
      std::deque<int> my_deque = {4, 6, 7};
      my_deque.push_front(2);
      my_deque.push_front(3);
      std::cout << my_deque.size() << std::endl;
      
      for(std::deque<int>::iterator itr = my_deque.begin();
          itr != my_deque.end();
          ++itr)
      {
         std::cout << *itr << " ";
      }
      std::cout << std::endl;

      std::cout << my_deque[1] << std::endl;
   }

   {
      /*
       *  List: Double linked list
       *  my_list1.splice(itr, my_list2, itr_a, itr_b);  // O(1)
       */
      std::list<int> my_list = {5, 2, 9};
      my_list.push_back(6);
      my_list.push_front(4);
      // my_list: {4, 5, 2, 9, 6}

      for(std::list<int>::iterator itr = my_list.begin();
          itr != my_list.end();
          ++itr)
      {
         std::cout << *itr << " ";
      }
      std::cout << std::endl;

      // my_itr -> 2
      std::list<int>::iterator my_itr = find(my_list.begin(), my_list.end(), 2);
      my_list.insert(my_itr, 8);
      // my_list: {4, 5, 8, 2, 9, 6}

      for(std::list<int>::iterator itr = my_list.begin();
          itr != my_list.end();
          ++itr)
      {
         std::cout << *itr << " ";
      }
      std::cout << std::endl;

      my_itr++;

      my_list.erase(my_itr);
            for(std::list<int>::iterator itr = my_list.begin();
          itr != my_list.end();
          ++itr)
      {
         std::cout << *itr << " ";
      }
      std::cout << std::endl;
   }

   {
      /*
       *  Array container: 
       *  int a[3] = {3, 4, 5};
       *  a.begin();
       *  a.end();
       *  a.size();
       *  a.swap();
       */
      // std::array container owns integers,
      std::array<int, 4> my_array = {3, 4, 5, 1};
      for ( auto iCnt : my_array)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::sort(my_array.begin(), my_array.end());

      std::cout << std::endl;
   }

   {
      /*
       * Associaive Container: 
       * set and multi-set;
       * The associative container is related to binary tree,
       * When an element is inserted/removed the binary tree is sorted. 
       * Always sored, default operator '<'.  No push_back()/push_front();
       * 
       * set
       * - no duplicates
       */
      std::set<int> my_set;
      my_set.insert(3);      // my_set: {3}
      my_set.insert(1);      // my_set: {1, 3}
      my_set.insert(7);      // my_set: {7, 1, 3}

      for ( auto iCnt : my_set)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::set<int>::iterator my_itertor;
      my_itertor = my_set.find(7);     // my_iterator points to 7
                                       // sequance iterators do not have find function member.

      // instantiate an container of type pair.
      // it owns an itrator of container type set owning intergers.
      // the second element of the pair is bool.
      std::pair<std::set<int>::iterator, bool> ret_pair;

      // insertion fails, since there is alreay a 3 in the set 
      // and duplication is not allowed.
      ret_pair = my_set.insert(3);

      for ( auto iCnt : my_set)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      if (ret_pair.second==false)          
         my_itertor=ret_pair.first;      // it now points to element 3;

      for ( auto iCnt : my_set)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      my_set.insert(my_itertor, 9);     // my_set: {7, 1, ,3 ,9}
                                        // my_iterator points to 3.
                                        // it is always sorted.

      for ( auto iCnt : my_set)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      my_set.erase(my_itertor);         // my_set: {7, 1, ,9}

      for ( auto iCnt : my_set)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      my_set.erase(7);                  // my_set: {1, ,9}

      for ( auto iCnt : my_set)    
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // multi-set is a set that allows duplicate items
      std::multiset<int> my_multiSet;

      // set/multi-set: values of the elements can not be modified.
      // *my_iterator = 10;   *my_iterator is read only.

      std::cout << std::endl;
   }

   {
      /*
       * Associaive Container: 
       * Map and Multi-map;
       * The associative container has a key and value pair,
       * Set and multi-set are always sorted according to their values.
       * Does not allow values with duplicate keys.
       * map
       * - no duplicates
       */
      // this is container that own key,value pair of char and integer.
      std::map<char, int> my_map;
      my_map.insert (std::pair<char, int>('z', 200));
      my_map.insert (std::pair<char, int>('a', 100));

      for ( auto iCnt : my_map)    
         std::cout << iCnt.first 
                   << " " 
                   << iCnt.second
                   << " ";
      std::cout << std::endl;

      std::map<char, int>::iterator my_iterator = my_map.begin();
      my_map.insert(my_iterator, std::pair<char, int>('b', 300));

      my_iterator = my_map.find('z');

      for ( my_iterator=my_map.begin();
            my_iterator != my_map.end();
            my_iterator++)
      {
         std::cout << (*my_iterator).first 
                   << " => " 
                   << (*my_iterator).second 
                   << std::endl;
      }

      // multi-map is just like a map however it allows duplicate keys.
      std::multimap<char, int> my_multiMap;
   
      std::cout << std::endl;
   }

   {
      /*
       * Unordered Associaive Container (C++ 11): 
       * Unordered set and multi-set
       * Unordered Map and Multi-map;
       * Link list of Hash Table and Hash Funtions.
       */
      
      /*
       * Unordered Set
       */
      std::unordered_set<std::string> my_unorderedSet = {"Red", "Green", "Blue"};
      std::unordered_set<std::string>::const_iterator my_iterator = my_unorderedSet.find("Green");

      for ( auto iCnt : my_unorderedSet)    
         std::cout << iCnt
                   << " ";
      std::cout << std::endl;

      if (my_iterator != my_unorderedSet.end())
         std::cout << *my_iterator << std::endl;
      
      my_unorderedSet.insert("Yellow");

      for ( auto iCnt : my_unorderedSet)    
         std::cout << iCnt
                   << " ";
      std::cout << std::endl;

      std::vector<std::string> my_vector = {"Purple", "Pink"};

      my_unorderedSet.insert(my_vector.begin(), my_vector.end());

      for ( auto iCnt : my_unorderedSet)    
         std::cout << iCnt
                   << " ";
      std::cout << std::endl;

      //Hash Table Sepcific APIs
      std::cout << "load_factor = " << my_unorderedSet.load_factor() << std::endl;
      std::string x = "Red";
      std::cout << x << " is in a bucket #" << my_unorderedSet.bucket(x) << std::endl;
      std::cout << "Total Bucket #" << my_unorderedSet.bucket_count() << std::endl;

      // hash collision => many items are inserted in same bucket.  All elements are
      // inserted in the same bucket.

      std::cout << std::endl;
   }
   
   {
      /*
       * Associatd Array
       * - map and unordered map.
       */

      std::unordered_map<char, std::string> day = {{'S', "Sunday"}, {'M', "Monday"}};
      std::unordered_map<char, std::string>::iterator my_iterator;

       for ( my_iterator=day.begin();
            my_iterator != day.end();
            my_iterator++)
      {
         std::cout << (*my_iterator).first
                   << " => " 
                   << (*my_iterator).second 
                   << std::endl;
      }
      std::cout << std::endl;

      std::cout << day['S'] << std::endl;          // no range check.
      std::cout << day.at('S') << std::endl;       // has range check.

       for ( my_iterator=day.begin();
            my_iterator != day.end();
            my_iterator++)
      {
         std::cout << (*my_iterator).first
                   << " => " 
                   << (*my_iterator).second 
                   << std::endl;
      }
      std::cout << std::endl;

      std::vector<int> my_vector = {1, 2, 3};
      // my_vector[5] = 6;                            // compiler error
      day['W'] = "Wednesday";                         // inserting 'W'
      day.insert(std::make_pair('F', "Friday"));      // inserting 'F'

       for ( my_iterator=day.begin();
            my_iterator != day.end();
            my_iterator++)
      {
         std::cout << (*my_iterator).first
                   << " => " 
                   << (*my_iterator).second 
                   << std::endl;
      }
      std::cout << std::endl;


      day.insert(std::make_pair('M', "MONDAY"));

       for ( my_iterator=day.begin();
            my_iterator != day.end();
            my_iterator++)
      {
         std::cout << (*my_iterator).first
                   << " => " 
                   << (*my_iterator).second 
                   << std::endl;
      }
      std::cout << std::endl;

      day['M'] = "MONDAY";

       for ( my_iterator=day.begin();
            my_iterator != day.end();
            my_iterator++)
      {
         std::cout << (*my_iterator).first
                   << " => " 
                   << (*my_iterator).second 
                   << std::endl;
      }
      std::cout << std::endl;

      std::cout << std::endl;

   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}