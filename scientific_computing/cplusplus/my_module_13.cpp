#include <cstdlib>
#include <iostream>     // std::cin, std::cout
#include <string>       // std::string
#include <fstream>      // std::ofstream
#include <iterator>     // std::istreambuf_iterator
#include <sstream>      // std::stringstream
#include <cstring>      // memset
#include <algorithm>    // count
#include <vector>       // vector
#include <initializer_list> // Initializer list
#include <assert.h>         // assert
#include <memory>           //shared pointer.

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_09.cpp
 */

// Scientific Computing Library
namespace scl
{
   class CPerson
   {
    private:
       // ptr_name_ is a pointer object of type std::string.
       // The ptr_name_ is an address 0x1234, *ptr_name_ is the
       // content and & ptr_name_ is another address.
       std::string * ptr_name_;
    public:
      CPerson() {}

      // CPerson construcor taking a string object as an input.
      // the data member 
      CPerson(std::string name) : ptr_name_(new std::string(name)) 
      {
        std::cout << "CPerson " << *ptr_name_ << std::endl;
      }

      // CPerson(const CPerson&) = delete;  //The copy constructor is deleted.
        
      ~CPerson() 
      { 
        std::cout << "CPerson deleted ..." << *ptr_name_ << std::endl;
        delete ptr_name_;
        ptr_name_ = nullptr;
      }
      
      void print_name() 
      { 
         std::cout << *ptr_name_ << std::endl;
      }
   };


   class CPerson_sharedPtr
   {
    private:
       // ptr_name_ is a pointer object of type std::string.
       // The ptr_name_ is an address 0x1234, *ptr_name_ is the
       // content and & ptr_name_ is another address.
       std::unique_ptr<std::string> ptr_name_;
    public:
      CPerson_sharedPtr() {}
      ~CPerson_sharedPtr() {}
      // move constructor
      CPerson_sharedPtr(CPerson_sharedPtr &&) = default;

      // CPerson construcor taking a string object as an input.
      // the data member 
      CPerson_sharedPtr(std::string name) : ptr_name_(new std::string(name)) 
      {
        std::cout << "CPerson_sharedPtr " << *ptr_name_ << std::endl;
      }

      // CPerson_sharedPtr(const CPerson_sharedPtr&) = delete;  //The copy constructor is deleted.

    //   Since using shared pointer, do not need the destructor anymore.         
    //   ~CPerson_sharedPtr() 
    //   { 
    //     std::cout << "CPerson_sharedPtr deleted ..." << *ptr_name_ << std::endl;
    //     delete ptr_name_;
    //     ptr_name_ = nullptr;
    //   }
      
      void print_name() 
      { 
         std::cout << *ptr_name_ << std::endl;
      }

      int return_useCount()
      {
        return 1;
      }
   };


}
/*
 * C++ 03 solution
 * 1. define copy constructor and copy assignment operator
 * 2. delete copy constructor and copy assignment operator.
 */

/*
 * C++ features;
 * 1. keyword 'delete' to delete a function.
 * 2. emplace_back() construct the object in place (in th space allocated to the vector) 
 *    It is used instead of push_back();
 * 3. std::shared_ptr
 * 4. std::unique_ptr
 * 5. move function in instruction 'persons.push_back(std::move(person));'
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;

   {
   // vector container owning the CPerson class
   std::vector<scl::CPerson> persons;

   // push back an instance of CPerson 'george' object in the vector container.
   // persons.push_back(scl::CPerson("George"));   This failed, since an object of CPerson is pushed back.
   // This works since the emplace back pushes a string "george" and the object is created.
   // Construct an object emplace (to put in place or position) in a container (vector<>)
   // This will construct the object in place (in th space allocated to the vector) 
   persons.emplace_back("George");

   // person.front() return an instance of the CPerson 'George'.
   // the front method return a refereance to the first character.
   persons.front().print_name();

   std::cout << "GoodBye" << std::endl;
   }  

   {
        std::cout << "Hello world" << std::endl;
        std::vector<scl::CPerson_sharedPtr> persons;
        scl::CPerson_sharedPtr person("Steve");
        // vector container owning the CPerson class
        persons.push_back(std::move(person));
        persons.front().print_name();
        int iCnt = persons.front().return_useCount();
        std::cout << iCnt << std::endl;
        std::cout << "GoodBye" << std::endl;
   }

   return 0;
}