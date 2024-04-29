#include <cstdlib>
#include <regex>      // use regex_math
#include <iostream>   // std::cin and std::cout
#include <string>     // string.
#include <chrono>     // time and date
#include <random>     // Random Number Generation.
#include <tuple>      // tuple

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_17.cpp
 */

/*
 * const
 * 1. it is a compiled time constraint that an object can not be modified.
 * 
 */

// Scientific Computing Library
namespace scl
{
   //constant used with functions.
   class CDog
   {
   private:
      int age_;
      std::string name_;
   public:
      CDog() 
      {
         age_ = 3;
         name_ = "dummy";
      }

      // constant parameters
      // takes the referance of some_age.
      void set_age(const int& some_age)
      {
         age_ = some_age;
         // some_age++;
      }
      void set_age2(const int some_age)
      {
         age_ = some_age;
         // some_age++;
      }

      // constant return value;
      const std::string& get_name()
      {
         return name_;
      }
      
      // constant function: none of the member varialbes of the CDog does not change due 
      // to this function member.  It fails to compile if a member variable is modified
      // within the sope of this function or a method is invoked where a member variable
      // is invoked (changing the the class's member variable.)
      void print_dogName() const 
      {
         std::cout << "constant: " << name_ << std::endl;
      }

      void print_dogName()
      {
         std::cout << "none - constant: " << name_ << std::endl;
      }
   };
 
   class CBigArray
   {
   private:
      std::vector<int> vector;  //huge vector
      int j = 0;
      int * const access_counter = &j;
      mutable int mutable_accessCounter;
      int * vector_ptr = &j;

   public:
      CBigArray()
      {
         vector.push_back(0);
         vector.push_back(1);
         // access_counter = nullptr;
      }      
      int get_item(int index) const
      {
         (*access_counter) ++;
         mutable_accessCounter ++;
         // const_cast<CBigArray*>(this->access_counter++);
         return vector[index];
      }
      void set_vectorPtrItem(int index, int x)
      {
         // * means content at some position, vector_ptr is an l-value or location
         // of a memory block, and index is offset from this l-value;
         *(vector_ptr + index) = x;
         
         // these two lines are exactly the same.
         // vector_ptr[index] = x;
         // *(vector_ptr + index) = x;

      }

   };


}

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;

{
   const int i = 9;
   int j = 10;
   //i = 6 will fail since i is a constant.

   /*
    * if const is on the left of a *, the data or content is a constant.
    * if const is on the right of a *, the pointer variable is a constant.
    */

   // data is constant, pointer is not constant.  *p1 has a value like 5, 6,7
   // the p1 is pointting to a location in memory block l-value.
   const int *p1 = &i;

   // *p1 = 5;  will recieve an error message 'read-only' location 

   // the pointer variable p1 which has values like 0xD42C can be 
   // incremented to 0xD430.
   std::cout << (p1)  << " " << (++p1) << std::endl;

   // data is not constant, pointer is constant.  *p2 has a value like 5, 6,7
   // and these values cn change.  The p2 is pointting to a location in memory 
   // block l-value and that value is constant.
   int* const p2 = &j; 

   // Data is a constant and pointer variable is a constant.
   const int * const p3 = &j;   

   int const *p4 = &i;
   const int *p5 = &i;
}

{
   // the data or r-value is a constant.
   const int i = 9;
   std::cout << i << std::endl;
   // i = 6 will produce compiler issues.
   const_cast<int&>(i) = 6;
   std::cout << i << std::endl;

   int j = 10;
   static_cast<const int&>(j);
   std::cout << "Good Bye" << std::endl;
   // static_cast<const int&>(j) = 7; will fail compilation, since the opertor static_cast<const int&>(j) 
   // will make int variable j a constant.
}
   {
   /*
    * Why use const;
    * a. Guard againts inadvertent write to the variable.
    * b. Self documenting.
    * c. Enable compiler to do more optimization, making code tighter.
    * d. const means the variable can be put in ROM.
    */
     scl::CDog my_dog;
     int i = 9;
     my_dog.set_age(i);
     std::cout << i << std::endl;

     my_dog.set_age2(9);
     std::cout << i << std::endl;

     const std::string& temp_string = my_dog.get_name();
     std::cout << temp_string << std::endl;

     my_dog.print_dogName();

     const scl::CDog my_constantDog;
     my_constantDog.print_dogName();

     std::cout << "Good Bye" << std::endl;
   }

   {
      scl::CBigArray my_bigArray;
      std::cout << my_bigArray.get_item(0) << std::endl;
      std::cout << my_bigArray.get_item(1) << std::endl;

      std::cout << "Good Bye" << std::endl;
   }
   std::cout << "Good Bye" << std::endl;
   return 0;
}