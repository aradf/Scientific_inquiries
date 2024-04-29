#include <cstdlib>
#include <regex>      // use regex_math
#include <iostream>   // std::cin and std::cout
#include <string>     // string.
#include <chrono>     // time and date
#include <random>     // Random Number Generation.
#include <tuple>      // tuple

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_19.cpp
 */

// Scientific Computing Library
namespace scl
{
class collar
{
private:
   std::string name_;
public:
   collar() {}
   collar(std::string name) : name_(name) {}
};

class dog
{
private:
   // the ptr_collar is an pointer object with the values like 0x1234
   // &ptr_collar is 0xABCD and *ptr_color is the r-value, temporary,
   // content or data on location 0x1234.
   collar * ptr_collar_;
public:
   // Solution 1.
   // overloading the assignment operator (self-assignment)
   // const means the data is constant, however the referance is not
   // other is an address 0x1234.
   // dog& operator=(const dog& other)
   // {
   //    if (this == &other)
   //    {
   //       return *this;
   //    }

   //    collar * ptr_originalCollar = ptr_collar_;
   //    ptr_collar_ = new collar( *other.ptr_collar_ );
   //    delete ptr_originalCollar;
   //    ptr_originalCollar = nullptr;
     
   //    return *this;
   // }

   // Solution 2. delegatioin
   dog& operator=(const dog& other)
   {
      // member by member copying of collars or call collar's operator=
      *ptr_collar_ = *other.ptr_collar_;

      return *this;
   }
};

class CDog
{
private:
   std::string dog_name_;
public:
   CDog(std::string dog_name) : dog_name_(dog_name) {}
   void bark() { std::cout << dog_name_ << " barked ..." << std::endl;}

};

class Cat
{
private:
   std::string name_;
public:
   Cat(char* name) 
   {
      std::cout << "Constructing Cat " << name << std::endl;
      name_ = name;
   }
   void meow() { std::cout << name_ << "meowed ..." << std::endl;}

};


}

/*
 * Handle self-assignment in operator=
 * Operator overload: exploits people's intuition and reduce their learning curve
 */

/*
 * Initialization issue
 * - A subtle problem that can crash yoru program.
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   scl::CDog dog("Gunner");
   dog.bark();

   std::cout << "Good Bye" << std::endl;
   return 0;
}