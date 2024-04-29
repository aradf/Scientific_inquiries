#include <cstdlib>
#include <iostream>
#include "my_pair.h"
#include "sequence.h"

/*
 * g++ -g -o ./run_compute my_module_02.cpp
 */

template <class T> 
T add(T a, T b)
{
    return a + b;
}

// The return value is return_type could be float or int.
template <class return_type, class arg_type> 
return_type cast(arg_type x)
{
   return (return_type)x;
}

template <typename T>
void print_data(T data)
{
    std::cout << data << std::endl;
}

template <class T>
T get_max(T a, T b)
{
    T result;
    result = (a>b) ? a : b;
    return result;
}

int main(int argc, char** argv)
{
    std::cout << "My Module" << std::endl;
    int x = add<int>(1, 2);

    double y = add<double>(3.3, 3.3);

    std::cout << x << std::endl;
    std::cout << y << std::endl;
    
    int c = cast<long>(10.1);
    float d = cast<float>(10.1);    
    std::cout << c << std::endl;
    std::cout << d << std::endl;    
   
    double data = 1.1;
    print_data<int>(data);
    print_data<double>(data);

    long l = 10, m = 11;
    std::cout << get_max<long>(l, m) << std::endl;

    mypair<int> pair_instance(l,m);
    std::cout <<  pair_instance.get_max() << std::endl; 

    mysequence<int, 5> sequence_instance;
    mysequence<double, 10> sequence_secondInstance;
    sequence_instance.set_member(0, 100);
    sequence_secondInstance.set_member(3, 500);

    std::cout << sequence_secondInstance.get_member(3) << std::endl;
    

    return 0;
}