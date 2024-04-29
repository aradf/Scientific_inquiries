/*
 * STL Headers
 */
#include <cstdlib>
#include <iostream>   // std::cin and std::cout
#include <string>     // std::string
#include <deque>      // deque

/*
 * boost Headers
 * using boost::lexical_cast;
 * using boost::bad_lexical_cast;
 */
#include "boost/lexical_cast.hpp"    // lexical_cast
#include "boost/variant.hpp"         // variant
#include "boost/any.hpp"             // any
#include "boost/optional.hpp"        // optional
#include "boost/array.hpp"           // arrays

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_30.cpp
 */

// Scientific Computing Library
namespace scl
{
   // stl template deque containing chars.
   std::deque<char> queue;

   char get_asyncData()
   {
      if (!queue.empty())
      {
         return queue.back();
      }else
         return '\0';      // this is a valid char.
   }

   // return a boost template optional containing a char.
   boost::optional<char> get_async_data()
   {
      if (!queue.empty())
      {
         return boost::optional<char>( queue.back() );
      }else
         return boost::optional<char>();      // this is a valid char.
   }
}  // end: namespace scl

/*
 * Convert from string
 * atof Convert string to double (function)
 * atoi Convert string to integer (function)
 * atol Convert string to long (function)
 * atoll Convert string to long long integer (function)
 * strtod Convert string to double (function)
 * strtof Convert string to float (function)
 * strtol Convert string to long integer (function)
 * sscant()
 * ....
 * 
 * Convert to string
 * string to stream strm;
 * strm << int_val;
 * string s = strm.str();
 * sprintf()
 * itoa  // non-standard
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;

   {
      int s = 23;
      // stl class type string is boost template lexical cast containing a stl class type string
      // initialzied with integer s.  
      std::string str = boost::lexical_cast<std::string>(s);

      // the stl class type string is concatinted with boost template 'lexical_cast' containing
      // stl class type string initialized with char 'A' and concatinated wit boost template
      // 'lexical_cast' containing stl class type string initiazed with float 34.5
      str = "Message: " + boost::lexical_cast<std::string>('A') + boost::lexical_cast<std::string>(34.5);
      std::cout << str << std::endl;

      // stl template array containing char and int64; boost template lecixal_cast containing
      // stl template array containing char and int64 initialized 28.
      std::array<char, 64> msg = boost::lexical_cast< std::array<char, 64> >(28);

      s = boost::lexical_cast<int>("3456");
      try
      {
         // the boost template contains an integer and float is passed.
         s = boost::lexical_cast<int>("56.78");
      }
      catch(boost::bad_lexical_cast& e)
      {
         std::cout << "Boost Exception Caught: " << e.what() << std::endl;
      }
      
      try
      {
         // the boost template contains an integer and none integer has passed.
         s = boost::lexical_cast<int>("3456yu");
      }
      catch(boost::bad_lexical_cast& e)
      {
         std::cout << "Boost Exception Caught: " << e.what() << std::endl;
      }
      
      // boost template lexical_cast containing integer; the first four char
      // are passed on.
      s = boost::lexical_cast<int>("3456yu", 4);
      std::cout << "Good Bye" << std::endl;
   }

   {
      // the i and f shared memory.
      union  
      {
         int i;
         float f;
      }u;
      u.i = 34;
      u.f = 2.3;   //u.i is over-written.
      
      /* Will not compile, since string, integers and float are not the same.
         union  
         {
            int i;
            std::string s;
            float f;
         }u;
      */
      // boost template variant containing integers, stl strings.
      // u1 and u2 can contain an integer or stl string not both 
      // at the same time.
      boost::variant<int, std::string> u1, u2;
      u1 = 2;          // integer not std::string.
      u2 = "hello";    // std::string not an integer.
      std::cout << u1 << " " << u2 << std::endl;

      // u1 = y1 * 2;  // will not compile operator * is not overloaded for boost template variant.
      
      // boost template get containing integers initialized u1.
      u1 = boost::get<int>(u1) * 2;
      
      // std::string st = boost::get<std::string>(u1);     // throws an exception

      u1 = "good";     // u1 becomes a std::string
      u1 = 32;         // u1 becomes an integer again.

      // boost template variant containing intgers and stl class type std::string.
      boost::variant<int, std::string> u3;
      std::cout << u3 << std::endl;

      // Using a vistor class:  Declar a class as a public child of 
      // boost template 'static_visitor'
      class DoubleVisitor : public boost::static_visitor<>
      {
         public:
         // over-load the () operator with referance of an
         // intger passed on.
         void operator() (int& i ) const
         {
            i += i;
         }
         // over-load the () operatorl with a stl class type 'string'
         // passed on.
         void operator() (std::string& str) const
         {
            str += str;
         }
      };
      // u1 variable is a boost template variant containing an integer or stl class type string.
      // boost::variant<int, std::string>
      u1 = 2;

      // boost algorithm 'apply_visitor' operating on a class inherited from 'static_visitor'
      boost::apply_visitor( DoubleVisitor(), u1);  // u1 becomes 4.
      std::cout << u1 << std::endl;
      
      u2 = "hello";
      // boost algorithm 'apply_visitor' operating on a class inherited from 'satatic_visitor'
      boost::apply_visitor( DoubleVisitor(), u2);  // u2 becomes 'hellohello'.
      std::cout << u2 << std::endl;

      // stl template vector containing a boost template variant containing an integer and/or 
      // stl class type std::string.  
      std::vector< boost::variant<int, std::string> > vector;
      vector.push_back("good");
      vector.push_back(23);
      vector.push_back("bad");
      vector.push_back(24);
      DoubleVisitor my_doubleVisitor;
      for (auto x : vector)
      {
         boost::apply_visitor(my_doubleVisitor, x);
         std::cout << x << " ";
      }
      std::cout << "Good Bye" << std::endl;
   }

   {
      // boost class type any
      boost::any x,y,z;
      x = std::string("hello");
      x = 2.3;
      y = 'z';
      z = std::vector<int>();      // uses dynamic storage vs stack storage.

      // boost template any_cast containing char.
      // will not compile; any cast returns a ocpy of y's data.
      std::cout << boost::any_cast<char>(y) << std::endl; 
      // boost template 'any_cast' containing double.  
      std::cout << boost::any_cast<double>(x) << std::endl;

      if (x.empty())
         std::cout<< "x is empty ..." << std::endl;

      if (x.type() == typeid(char))
         std::cout << "this is a char" << std::endl;

      try
      {
         std::cout << boost::any_cast<int>(x) << std::endl;
      }
      catch( boost::bad_any_cast& e)
      {
         // boost exception type 'bad_any_cast'
         std::cerr << e.what() << std::endl;;
      }

      try
      {
         std::cout << boost::any_cast<float>(x) << std::endl;
      }
      catch( boost::bad_any_cast& e)
      {
         // boost exception type 'bad_any_cast'
         std::cerr << e.what() << std::endl;;
      }

      boost::any_cast< std::vector<int> >(z).push_back(23);
      // int i = boost::any_cast< std::vector<int> >(z).back();    // crash since any_cast returns a copy and (z) is still empty.
 
      int i = 11;
      // boost class type any 
      boost::any p = &i;

      // ptr_int is a pointer variable.  ptr_int = 0x1234, *ptr_int is data/content/r-value
      // &ptr_int is an address 0xABCD.  boost template 'any_cast' containing integer pointers.
      int * ptr_int = boost::any_cast<int*>(p);

      std::cout << *ptr_int << std::endl;
      std::cout << (boost::any_cast<int*>(p)) << std::endl; 
      std::cout << *(boost::any_cast<int*>(p)) << std::endl; 

      // stl template vector containing 'boost::any' class type.
      std::vector< boost::any > m;
      m.push_back(2);
      m.push_back('a');
      m.push_back(p);
      m.push_back( boost::any() );

      struct Property
      {
         // stl class type string and boost class type any
         std::string name;
         boost::any value;
      };
 
      // stl template vector containin 'Property' structures.
      std::vector<Property> properties;
      std::cout << "Good Bye" << std::endl;
   }

   {
      // boost template variant containiing a nullptr_t type and char.
      boost::variant<nullptr_t, char> v;

      // boost template optinal containing a char.
      boost::optional<char> op;   // op is optional and not initialized, no char is constructed.
      op = 'A';  // op contains 'A';

      op = scl::get_async_data();
      if (!op)
         std::cout << "No data is available ..." << std::endl;
      else
      {
         // only when the op is NOT empty, it will crash if op is empty.
         std::cout << "op contains: " << op.get() << std::endl;  
         std::cout << "op contains: " << *op << std::endl;  
      }
      
      // reset up to uninitialzied state
      op.reset();
      // return z if op is empty.
      std::cout << op.get_value_or('z') << std::endl;

      // returns null if op is empty.
      char * p = op.get_ptr();  

      struct A
      {
         std::string name;
         int value;
      };

      A a; 
      
      // constructor of A is not called, since opA0 is not initialized.
      boost::optional<A> opA0;

      // a is copy constructed into opA
      boost::optional<A> opA(a);
      
      std::cout << opA->name << " " << opA->value << std::endl;

      // boost template optional containing pointers to structure A.
      boost::optional<A*> opAP(&a);
      (*opAP)->name = "Bob";
      (*opAP)->value = 50;

      // boost template optional containing refereance to structure A.
      boost::optional<A&> opAR(a);
      (opAR)->name = "Bob";        //this changes a.name;
      (opAR)->value = 50;

      // relational opeator
      boost::optional<int> i1(1);
      boost::optional<int> i2(9);

      if (i1 < i2)   std::cout << "i2 is bigger" << std::endl;

      std::cout << "Good Bye" << std::endl;
   }

   {
      // define a boost array: boost::array<type, size> name_of_array;
      boost::array<std::string, 3> array_strings;

      // initialize
      array_strings = {"Boost", "C++", "Array"};

      // sort an array.
      // stl alrogithem sort operating on range of data in array_strings.
      std::sort(array_strings.begin(), array_strings.end());

      for (auto iCnt : array_strings) std::cout << iCnt << " ";
      std::cout << "Good Bye" << std::endl;   

      for (const std::string &str : array_strings) std::cout << str << " ";
      std::cout << "Good Bye" << std::endl;   

      // get size of boost template array.
      std::cout << "Size: " << array_strings.size() << std::endl;

      // get first element of boost::array or  boost template array.
      std::cout << "First Element: " << array_strings.front() << std::endl;

      // get first element of boost::array or  boost template array.
      std::cout << "Last Element: " << array_strings.back() << std::endl;
   }

   {
      // Boost Graph Library (BGL)
      std::cout << "Good Bye" << std::endl;
   }
   
   std::cout << "Good Bye" << std::endl;
   return 0;
}