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
class dog
{
private:
   std::string dog_name_;
public: 
   dog() {}
   ~dog() {}

   // Not explicit, this constructor is also a type converter that takes a std::string object to a 
   // scl::dog object.
   dog(std::string name) : dog_name_(name) 
   {
       std::cout << "Dog constructor: " << dog_name_ << std::endl;
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
   dog(const dog& other)
   {
       dog_name_ = other.dog_name_;
   }
 
   /* Copy Assignment operator: */
    dog& operator=(const dog& other)
    {
        if (this != &other)
        {
            this->dog_name_ = other.dog_name_;
        }
        return *this;
    }
};

/*
 * Implicit type conversion is useful when creating numerical types of class, 
 * such as a ratinal class.
 */

class Rational
{
public:
   std::string name_;
   int numerator_;
   int denominator_;
public:
   Rational() {}
   // since this constructor does not have implicit called out in from of it,
   // this is a constructor, explicit type converter and implicit type converter.
   Rational(std::string name) : name_(name) 
   {
       std::cout << "Rational Created .." << name_ << std::endl;
   }   
   Rational(int numerator = 0, int denominator = 1) : numerator_(numerator), denominator_(denominator)
   {
       this->name_ = "nameless";
       std::cout << "Rational Created .." 
                 << name_ 
                 << " " 
                 << this->numerator_
                 << " " 
                 << this->denominator_
                 << std::endl;
   }
   ~Rational() 
   {
       std::cout << "Rational deleted .." 
                 << this->name_ 
                 << " "
                 << this->numerator_ 
                 << " "
                 << this->denominator_
                 << std::endl;     
   }
   /* Copy Constructor */
   Rational(const Rational& other)
   {
       name_ = other.name_;
   }
   /* Copy Assignment operator: */
   Rational& operator=(const Rational& other)
   {
       if (this != &other)
       {
           this->name_ = other.name_;
       }
       return *this;
   }
   
//    // For the instruction: scl::Rational r3 = r1 * 2 ;
//    const Rational operator*(const Rational& rhs)
//    {
//      std::cout << "Returned values " 
//                << numerator_*rhs.numerator_
//                << " "
//                << denominator_*rhs.denominator_
//                << std::endl;
//      return Rational(numerator_*rhs.numerator_, 
//                      denominator_*rhs.denominator_);
//    }

//     // Next line of instruction will not compile, since the compiler does not have
//     // type conversion from rational to interger that is 
//     // an operator* (operand tyeps are 'int' and 'scl::rational')
//     // for the instruction: scl::Rational r3 = 3 * r1;
//     operator int () const 
//     { 
//         return (this->numerator_ / this->denominator_) ; 
//     }
};

//gloabal multipler for instruction: Rational r3 = 3 * r1;
const Rational operator*(const Rational& lhs, const Rational& rhs)
{
    return ( Rational(lhs.numerator_*rhs.numerator_, 
                      lhs.denominator_*rhs.denominator_));
}

}

/*
 * User defined implicit type conversion.
 * Categories of Type conversions
 *                               implicit    explicit
 * Standard Type Conversion      A           B
 * User defined type conversion  C           D
 *                                        casting
 * Category A:  Implicit standandard type conversion.
 *              type conversions between Integers, Doubles, etc ...
 *              user defined type conversion between user defined classes.
 * 
*/

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   {
      // Category A:  Implicit standandard type conversion.
      char my_char = 'a';
      int my_int  = my_char;               // integral promotion
      char * ptr_char = 0;                 // null pointer initialization
      
      // void f(int my_int);
      // f(my_char);

      scl::dog * ptr_dog = new scl::dog("Gunner");  //pointer type conversion.
   }
 
   {
      /*
       * Category C: Implicit user define type conversion.
       * Defined inside class (user defined type)
       * There are two methods to define implicit user defined type conversion.
       * Method 1: Use constructor that can accept a single parameter.
       *           - convert other types of object into your class
       * Method 2: Use the type conversion function
       *           - convert an instance of your class into other types.
       * 
       */
       std::string dog_name = "Bob";
       scl::dog dog1 = dog_name;
       std::cout << "My name is " << dog1.get_name() << std::endl;
       
       // use the type conversion function to convert an instance of
       // your class to othe types (std::string)
       std::cout << "My name is " << (std::string)dog1 << std::endl;
       std::string string_dog2 = (std::string)dog1;

       /*
        * Principle: make interface easy to use correctly and hard to 
        * use incorrectly.  How hard is enough? Idally, uncompilable.
        * 
        * General guidlines
        * 1. Avoid defining seemingly unexpected conversion.
        * 2. Avoid defining two-way implicit conversion.
        */

       std::cout << "Good Bye" << std::endl;
   }

   {
        scl::Rational r1 = 23;
        // the value 2 is passed on to he Rational construcor as r-value or tempertory data.
        // once the '*' overloaded operator is invoked, the r-value or temperary data for
        // interger two is deleted.
        scl::Rational r2 = r1 * 2;

        // Next line of instruction will not compile, since the compiler does not have
        // type conversion from rational to interger that is 
        // an operator* (operand tyeps are 'int' and 'scl::rational')
        scl::Rational r3 = 3 * r1;

        std::cout << "Good Bye" << std::endl;
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}