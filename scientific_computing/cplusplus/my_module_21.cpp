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
class CParentDog
{
private:
   std::string dog_name_;
public: 
   CParentDog() 
   {
     this->dog_name_ = "nameless";
   }
   ~CParentDog() 
   {
    std::cout << "CParentDog destructor: " << dog_name_ << std::endl;
   }

   // Not explicit, this constructor is also a type converter that takes a std::string object to a 
   // scl::dog object.
   CParentDog(std::string name) : dog_name_(name) 
   {
       std::cout << "CParentDog constructor: " << dog_name_ << std::endl;
   }
   // if you only want to define a constructor, and no implicit type
   // conversion, always put "explicit" before the constructor to
   // avoid inadvertent type conversion.
   // Adding 'explicit' infront of the constructor dog(std::string name) : dog_name_(name) {}
   // will tell the compiler that this constructor is explicit constructor.

   std::string get_name() 
   {
    return dog_name_;
   }

   // type conversion function from an instance of dog object to a std::string object.
   operator std::string () const 
   {
    return dog_name_;
   }

   /* Copy Constructor */
   CParentDog(const CParentDog& other)
   {
       dog_name_ = other.dog_name_;
   }
 
   /* Copy Assignment operator: */
    CParentDog& operator=(const CParentDog& other)
    {
        if (this != &other)
        {
            this->dog_name_ = other.dog_name_;
        }
        return *this;
    }
    virtual void bark() const  
    {
        std::cout << "I am just a CParentDog ..." << dog_name_ << std::endl; 
        // *this a r-value or temporary data (content ot this pointer) has become constant
        // Compiler error: dog_name_ = "Henry".
        // the const_cast is a powerful hack to take away the constant property.
        const_cast<CParentDog*>(this)->dog_name_ = "Henry";
    }
};   // end: class CParentDog

   class CYellowDog: public CParentDog
   {
    private:
       int age_;
    public:
       CYellowDog() 
       {
         std::cout << "CYellowDog born ..." << std::endl; 
         age_ = 0;
       }
       ~CYellowDog() 
       {
         std::cout << "CYellowDog destroyed ..." << std::endl; 
       }
       void bark() 
       {
         std::cout << "I am just a CYellowDog woof ..." << std::endl; 
         CParentDog::bark();
       }
   };   // end: CYellowDog

}  // end: namespace scl

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;

   {
        /*
         * All Casting Considered.
         * Type Conversion:
         * 1. Implicit Type Conversion.
         * 2. Explicit Type Conversion - casting
         */

        /*
         * 1. static_cast
         */
        int iCnt = 9;
        // perform static cast on a float variable type with int as an input.
        // it is ultimately a function that converts object from one type to another type.
        float fCnt = static_cast<float>(iCnt);
        
        //Type conversion needs to be defined.
        scl::CParentDog dog1 = static_cast<scl::CParentDog>(std::string("Bob"));
         
        // convert pointer/reference from one type to related type (down/up cast) 
        scl::CParentDog * ptr_dog = static_cast<scl::CParentDog*>(new scl::CYellowDog());
        std::cout << "Good Bye" << std::endl;        
   }

   {
        /*
         * 1. dynamic cast
         */
        scl::CParentDog * ptr_dog = new scl::CYellowDog();
        ptr_dog->bark();
        // (down cast)
        scl::CYellowDog * ptr_yellowDog = dynamic_cast<scl::CYellowDog *>(ptr_dog);
        ptr_yellowDog->bark();
        // a. It converts pointer/referance from one type to other type (down cast)
        // b. Run-time type check.  if success, ptr_dog = ptr_yellowDog, if fail, ptr_yellowDog =0 0
        // c. It requres the two types to be polymorphic (have virutal function.)
        std::cout << "Good Bye" << std::endl;
   }

   {
        /*
         * 3. const_char
         * only works on pointer/referance
         * Only wokks with the same type.
         * cast away constness of the object being pointed to.
         */
        const char * str = "Hello, world.";
        char * modifiable = const_cast<char*>(str);

        std::cout << "Good Bye" << std::endl;
   }

   {
        /*
         * 3. reinterpret_cast
         * re-interpret the bits of the object pointed to the ultimate cast that can
         * cast one ponter to any other type of pointer.
         * In this example 51110980 is a location or address in memory block.
         * The reinterpret_cast takes the r-value temporary data (content) at that location
         * and reinterpret it to a CParentDog.
         */
        long p = 51110980;
        scl::CParentDog * some_dog = reinterpret_cast<scl::CParentDog*>(p);
        std::cout << "Good Bye" << std::endl;
   }

   {
        /*
         * C-style casting.
         */ 
        short a = 2000;
        int i = (int)a;
        int j = int(a);

        /*
         * Generally C++ style of casts are prefered over the C-style casting.
         * 1. Easier to identify in the code.
         * 2. Less useage error. C++ style provides.
         *    a. Narrowly specified purpose of each cast, and 
         *    b. Run-time type check capability.
         */
   }

   {
        scl::CParentDog * ptr_dog = new scl::CParentDog();
        if (ptr_dog != nullptr)
          ptr_dog->bark();

        // static_cast vs dynamic_cast:  The static_cast does not perform run-time type check.
        // the ptr_yelloDog is not a nullptr and the results are undefined.  
        scl::CYellowDog * ptr_yellowDog = dynamic_cast<scl::CYellowDog*>(ptr_dog);
        // The ptr_yellowDog is a nullptr
        // This is a bug: scl::CParentDog * ptr_dog = new scl::CParentDog();
        // It sould be: scl::CParentDog * ptr_dog = new scl::CYellowDog();
        if (ptr_yellowDog != nullptr)
          ptr_yellowDog->bark();

        std::cout << "ptr_yellowDog: " << ptr_yellowDog << std::endl;
        std::cout << "ptr_dog: " << ptr_dog << std::endl;

        std::cout << "Good Bye" << std::endl;   
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}