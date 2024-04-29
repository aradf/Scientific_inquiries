#include <cstdlib>
#include <iostream>
#include <string.h>

/*
 * g++ -g -o ./run_compute hello.cpp
 */

int main(int argc, char** argv)
{
    std::cout << "Hello world" << std::endl;
    int age = 10;
    float float_age = 10.1;
    char charA = 'a';

    std::cout << "value of int" << age
              << "value of float" <<  float_age
              << "value of char "  << charA
              << std::endl;

    std::cout << "Defining Constant"
              << std::endl;

    const int MY_AGE = 5;

    std::cout << "Operators"
              << std::endl;

    int x = 20;
    x += 4;
 
    int data[5] = {1, 2, 3, 4, 5};
    std::cout << "Index in an array" 
              << std::endl;

    for (int iCnt = 0; iCnt < 5; iCnt ++)              
    {
        std::cout << data[iCnt] << " ";
    }
    std::cout << std::endl;

    int * ptr_data = nullptr;

    ptr_data = data;
    std::cout << ptr_data  
              << " "
              << *ptr_data
              << " "
              << &ptr_data;
    std::cout << std::endl;                            

    for (int iCnt = 0; iCnt < 5; iCnt ++)              
    {
        std::cout << ptr_data[iCnt] << " ";
    }
    std::cout << std::endl;
 
    struct STUDENT
    {
       char name[15];
       std::string department_name;
       float age;
       float height;
    };

    char name[15] = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd'};
    size_t byte_size = sizeof(name) * sizeof(char);
    
    STUDENT student;
    // memcpy(student.name, 
    //        name, 
    //        byte_size);
    strcpy(student.name, 
           name );

    student.department_name = "math";
    student.age = 10;
    student.height = 6;

    return 0;
}