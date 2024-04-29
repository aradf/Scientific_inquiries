#include <cstdlib>
#include <regex>      // use regex_math
#include <iostream>   // std::cin and std::cout
#include <string>     // string.
#include <chrono>     // time and date
#include <random>     // Random Number Generation.
#include <tuple>      // tuple

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_18.cpp
 */

/*
 * Compiler silently writes four functions if they are not explictly declared.
 * 1. Copy constructor.
 * 2. Copy Assignment Operator.
 * 3. Destructor.
 * 4. Default Constructor.
 */

/*
 * Summary of disallowing functions.
 * 1. C++ 11: f(0 = delete());
 * 2. C++ 03: Declare the function member to be private, and not define it.
 * 3. Private destructor: stay out of stack. 
 */

/*
 * Prevent Exceptions from leaving Destructors.
 */

// Scientific Computing Library
namespace scl
{
   class COpenFile
   {
   private:
      std::string file_name_;

   public:
      COpenFile()
      {
         std::cout << "COpenFile" << std::endl;
      }
      COpenFile(std::string file_name) : file_name_(file_name)
      {
         std::cout << "COpenFile: " << file_name_ << std::endl;
      }
      COpenFile( const COpenFile& other) = delete;          // delete the compiler generated copy constructor.
      COpenFile& operator=(const COpenFile& rhs) = delete;  // delete the compiler generated copy assignment operator.

      ~COpenFile()
      {
         std::cout << "~COpenFile: " << file_name_ << std::endl;
      }

      void destroy_me() {delete this;}
/*
      Make the copy constructor for class COpenFile  private function member to delete it.
      Make the copy assignment operator a private function member of COpenFile.
      private:
         COpenFile( const COpenFile& other);
         COpenFile& operator=(const COpenFile& rhs);
         void write_lines(std::string some_string);

*/

/*
      COpenFile( const COpenFile& other)
      {
         file_name_ = other.file_name_;
      }
*/      
   };

   class CDog
   {
   public:
      std::string name_;
      CDog(std::string name) : name_(name) { std::cout << name_ << " is born." << std::endl;    }
      ~CDog() 
      {
         /*
          * Solution 1: Destructor swallow the exception.
          */
         try
         {
            std::cout << name_ << " is destroyed ..." << std::endl; 
         }
         catch(const std::exception& e)
         {
            std::cerr << e.what() << '\n';
         }
         catch( ... )
         {
            std::cout << "All exceptions are caught ...";
         }
      }
      void prepae_toDestroy() { throw 20; }
      void bark() { std::cout << name_ << " Barked ..." << std::endl;   }
   };

   class CParentDog
   {
    public:
       CParentDog() 
       {
         std::cout << "CParentDog born ..." << std::endl; 
         bark();
       }
       virtual void bark() 
       {
         std::cout << "I am just a CParentDog ..." << std::endl; 
       }
       void see_cat() 
       {
         bark(); 
       }
       ~CParentDog() 
       {
         bark();
       }
   };

   class CYellowDog: public CParentDog
   {
    public:
       CYellowDog() 
       {
         std::cout << "CYellowDog born ..." << std::endl; 
       }
       virtual void bark() 
       {
         std::cout << "I am just a CYellowDog ..." << std::endl; 
         CParentDog::bark();
       }
   };
}

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
 
   {
      scl::COpenFile open_file1(std::string("Hello.txt"));
      // scl::COpenFile open_file2(open_file1);
   }

   {
      // a privte destructor can only be invoked on heap and 
      // can not be destroyed on stack.
      scl::COpenFile* open_file1 = new scl::COpenFile(std::string("Hello.txt"));
      open_file1->destroy_me();
   }

   {
      try
      {
         scl::CDog dog_one("Henry");
         scl::CDog dog_two("Bog");
         dog_one.bark();
         dog_two.bark();
         dog_one.prepae_toDestroy();
         dog_two.prepae_toDestroy();
      }
      catch(int e)
      { 
         // before the throw 20 is caught, the objects are deleted from stack and 
         // their memroy is available.
         std::cout << e << " is caught ..." << std::endl;
      }
      catch(const std::exception& e)
      {
         std::cerr << e.what() << '\n';
      }
   }

   {
      // calling virtual functions in constructor or destructor ...
      // when a CYelloDog is instantiated, it invokes the parents constructor.
      // Must avoid calling virutal functions in the middle of a constructor.
      // the constructor sould do as little as possible to put the object in 
      // a valid state.  It is not a good idea to invoked function member inside
      // of a destructor.
      scl::CYellowDog yellow_dog;
      yellow_dog.see_cat();
      std::cout << "Good Bye" << std::endl;
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}