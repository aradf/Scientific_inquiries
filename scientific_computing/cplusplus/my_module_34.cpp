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
public:
   virtual void bark() 
   { 
      std::cout << " name less can not bark ... " << std::endl;
   }
};

class YellowDog : public Dog
{
   std::string my_name_;
public:
   YellowDog( std::string name ) : my_name_(name)
   {
      std::cout << " Yellow Dog: " << my_name_ << std::endl;
   }
   virtual void bark() 
   { 
      std::cout << " Yellow Dog: " << my_name_ << " barks ..." << std::endl;
   }

};

}  // end: namespace scl

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;

   {
      std::deque<scl::Dog> my_dog;
      scl::YellowDog my_yellowDog("Gunner");

      my_dog.push_front(my_yellowDog);
      my_dog[0].bark();
   }

   {
      std::deque<scl::Dog * > my_dog;
      scl::YellowDog my_yellowDog("Gunner");

      my_dog.push_front(&my_yellowDog);
      my_dog[0]->bark();
   }

   {
      

   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}