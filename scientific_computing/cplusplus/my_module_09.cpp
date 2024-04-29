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

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_09.cpp
 */

// https://en.cppreference.com/w/cpp/language/move_assignment

// Scientific Computing Library
namespace scl
{
    // scientific Computing Enum
}

/*
 * c++ 03
 * 1. define construtor (generated only no consutructor is declared by user.)
 * 2. Copy Constructor (generated only no 5 and 6 declared by user.)
 * 3. Copy assignment operator (generated only if 5,6 is declared by user.)
 * 4. destructor
 * 
 * c++ 11
 * 5. Move constructor (generated only if 2,3,4,6 not declared by user)
 * 6. Move assignment operator (generated onlly if 2,3,4,5 not declared by user.)
 */
class CDog
{
public:
   int tag_;

public:
   CDog() {}
   ~CDog() {}

   /*
    * Copy Constructor:
    * The some_dog is an l-value reference object.  l-value is a location in the memory block.
    * where r-value or a temporary object can be stored.
    */
   CDog(const CDog& other)
   {
       tag_ = other.tag_;
   }

   /*
    * Copy Assignment operator:
    * 
    */
    CDog& operator=(const CDog& other)
    {
        if (this != &other)
        {
            tag_ = other.tag_;
        }
        return *this;
    }

    /*
     * Move Constructor; 
    */
    CDog(CDog &&) {}

    /*
     * Move assignment operator
    */
    CDog& operator=(CDog && other) 
    { 
       return *this;
    }
};


int main(int argc, char** argv)
{
    std::cout << "Hello world" << std::endl;
    CDog dog_one;
    CDog dog_two = dog_one;
    CDog dog_three(dog_one);
 
    std::cout << std::endl;
    return 0;
}