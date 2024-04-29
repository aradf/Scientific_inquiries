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
 * g++ -std=c++17 -g -o ./run_compute my_module_19.cpp
 */

// Scientific Computing Library
namespace scl
{

class X
{
   private:
   int number_;
   public:
   void operator()(std::string some_string)
   {
      std::cout << "Calling functor x with parameters " 
                << some_string 
                << std::endl;
   }

   // type conversion function: casting an instance of X with std::string
   operator std::string () const 
   { 
      return "X";
   }

   // Constructor for X
   X(int i) 
   {
      std::cout << "Init object of class X " << i << std::endl; 
      this->number_ = i;
   }

   // Constructor for X
   X() 
   {
      std::cout << "Init object of class X " << std::endl; 
      this->number_ = 0;
   }
};

void add2(int i)
{
   std::cout << i + 2 << std::endl;
}

// define a template function with class type integer,
template<int value>
void add_value(int i)
{
   std::cout << value + i 
             << std::endl;
}

// define a functor.
class CAddValue
{
   int value;
   public:
      CAddValue(int j) : value(j)
      {
         std::cout << "Functor CAddValue is invoked ... " 
                   << value
                   << std::endl;
      }
      // 
      void operator() (int i)
      {
         std::cout << i + value << std::endl;
      }

};

void add_value2(int i, int value)
{
   std::cout << value + i 
             << std::endl;
}

double pow(double first, double second)
{
   return std::pow(first, second);
}

bool need_copy(int x)
{
   return (x>20 || x<5);
}

class CLsb_less
{
   public:
   bool lsb_less(int x, int y)
   {
      return ( x%10 < y%10 );
   }
};


}  // end: namespace scl

/*
 * Function Objects (functors)
 * Example :
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   
   {
      scl::X instance_x;

      //calling functor X with parameter Hi
      instance_x("Hi") ;

      std::cout << (std::string)(instance_x)  << std::endl;

      std::cout << std::endl;
   }

   {
      /*
       * Benifits of functor:
       * 1. Smart function:  capabilities beyond operator();
       *    it can remember state.
       * 2. It can have its own type.
       */

      /*
       * Parameterized Function
       */
      scl::X(8)("Hi");

      std::vector<int> vector = { 2, 3, 4, 5};

      // take the elements of vector and apply add2 to it.
      std::for_each(vector.begin(), vector.end(), scl::add2);           // 4, 5, 6, 7

      // take the elements of the vector and apply template function.
      std::for_each(vector.begin(), vector.end(), scl::add_value<2>);   // 4, 5, 6, 7

      int x = 2;
      std::for_each(vector.begin(), vector.end(), scl::CAddValue(x));

      std::cout << std::endl;
   }

   {
      /*
       * Build-in Functors
       * less, greater, greater_euqal less_equal not_equal_to
       * logical_and logical_not logical_or
       * multiplies minus plus divide modulus negate
       */
      int x = std::multiplies<int>()(3,4);

      if (std::not_equal_to<int>()(x, 10))
         std::cout << x << std::endl;
      
      std::cout << std::endl;
   }

  {
      /*
       * Parameter binding
       */
      // declare a set object containing integers and initialized.
      std::set<int> my_set={2, 3, 4, 5};
      std::vector<int> my_vector;

      int x = std::multiplies<int>() (3,4);   // x = 3 * 4


      // multiply my_set's elements by 10 and save it in vector.
      // there is a standard functor named std::multiplies.
      // the std::transform applies the functor on every element of the my_set.
      // std::bind function means the std::placeholders::_1 is 

      std::transform(my_set.begin(), 
                     my_set.end(), 
                     std::back_inserter(my_vector), 
                     std::bind(std::multiplies<int>(),
                     std::placeholders::_1, 10));
      // First parameter of multiplies<int>() is substituted with my_st's element.
      for (auto iCnt: my_vector)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      std::for_each(my_vector.begin(), 
                    my_vector.end(), 
                    bind(scl::add_value2, 
                    std::placeholders::_1, 2));      

      // c++ 03: bind1st, bind2nd
      // c++ 11: bind

      std::cout << std::endl;
   }

   {
      /*
       * Convert regular function to functor
       */
      std::set<int> my_set = {3, 1, 25, 7, 12};
      std::deque<int> my_deque;

      //C++ 11
      // 'auto' return the proper data or object type.
      // 'std::function' is a template class function from standard template library
      // 'scl::pow' is a regular function that has become a functor my_functor.
      // note the first double is the return value, the (double, double) are signatures
      // of the scl::pow.
      auto my_functor = std::function<double (double, double)>(scl::pow);
      std::transform(my_set.begin(), 
                     my_set.end(), 
                     std::back_inserter(my_deque),
                     std::bind(my_functor, std::placeholders::_1, 2));
       
      // myset.bind(), myset.end() is the source,
      // std::back_inserter(my_deque) is the destination
      // std::bind(my_functor, std::placeholders::_1, 2) is the functor
      // my_deque: {1, 9, 49, 144, 625}
      for (auto iCnt : my_deque)
         std::cout << iCnt << " ";
      std::cout << std::endl;   
 
      //std::function is c++11
      //std::ptr_fun is c++03
    }

   {
      //template class set containing integers
      std::set<int> my_set = { 3, 1, 25, 7, 12};
      std::deque<int> my_deque, my_deque2, my_deque3;

      // Template class transform applies the functor to every element of my_set
      std::transform(my_set.begin(), 
                     my_set.end(), 
                     std::back_inserter(my_deque),
                     std::bind(std::logical_or<bool>(),
                     std::bind(std::greater<int>(), std::placeholders::_1, 20),
                     std::bind(std::less<int>(), std::placeholders::_1, 5)));

      
      for ( auto iCnt : my_deque)
         std::cout << iCnt << " ";

      std::cout << std::endl;
      

      auto my_needcopyFunctor = std::function<bool (int)>(scl::need_copy);
      std::transform(my_set.begin(), 
                     my_set.end(), 
                     std::back_inserter(my_deque2),
                     my_needcopyFunctor);

      for ( auto iCnt : my_deque2)
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // c++ 11 lambda function
      // the template class transform applies the functor to every element
      // of the my_set.  the template class back_inserter inserts in 
      // my_deque3.
      std::transform(my_set.begin(), 
                     my_set.end(),
                     std::back_inserter(my_deque3),
                     [](int x){return scl::need_copy(x); });

      std::cout << std::endl;
   }  
 
   {
      // lambda functions and functors are invoking a function member 
      // using pointers.

      // template class set contains integer.
      std::set<int> my_set = {3, 1, 25, 7, 12};

      for (auto iCnt : my_set )
         std::cout << iCnt << " ";
      std::cout << std::endl;

      // the template class set can take two sets of template parameters.
      std::set<int, std::less<int> > my_set1 = {3, 1, 25, 7, 12};

      for (auto iCnt : my_set1 )
         std::cout << iCnt << " ";
      std::cout << std::endl;

      class Lsb_less
      {
         public:
         bool operator()(const int& x, const int& y) const
         {
            return ( x%10 < y%10 );
         }
      };

      std::set<int, Lsb_less > my_set2 = {3, 1, 25, 7, 12};
      for (auto iCnt : my_set2)   std::cout << iCnt << " ";
      std::cout << std::endl;

      std::cout << std::endl;
   }


   std::cout << "Good Bye" << std::endl;
   return 0;
}