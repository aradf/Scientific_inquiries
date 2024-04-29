#include <cstdlib>
#include <iostream>
#include <string>

/*
 * g++ -g -o ./run_compute my_module_01.cpp
 */


int main(int argc, char** argv)
{
    std::cout << "My Module" << std::endl;

    // structure are used as data containers.
    struct person_t 
    {
       std::string name;
       unsigned age;
    };

    // Complex data structure.
    class CPerson
    {
        std::string name_;
        unsigned age_;
     public:
        CPerson() 
        {
            name_ = ""; 
            age_ = 0;
        };
        unsigned age() const 
        {
            return this->age_;
        }
        void set_age(unsigned a)
        {
            this->age_ = a;
        }
    };

    person_t * some_person = nullptr;
    CPerson * person_instance = new CPerson();
    
    person_instance->set_age ( 5 );
    std::cout << person_instance->age() << std::endl;
  
    return 0;
}