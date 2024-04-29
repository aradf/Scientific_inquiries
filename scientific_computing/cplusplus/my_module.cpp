#include <cstdlib>
#include <iostream>

/*
 * g++ -g -o ./run_compute my_module.cpp
 */

class CMath
{
public:
    int a;
    int b;

    CMath()
    {
        this->a = 0;
        this->b = 0;
    }
 
    CMath(int a, int b)
    {
        this->a = a;
        this->b = b;
    }
public:
    int sum_number( int a,
                    int b)
    {
        return (a + b);
    }                

    void print_out(int sum)
    {
        std::cout << "Sum " 
                << sum
                << std::endl;
    }
};

class A
{
public:
    int a;
    A()
    {
        this->a = 2;
    }
    void print()
    {
        std::cout << "class a" << std::endl;
    }
};

class B : public A
{
public:
    int b;
    B()
    {
        this->b = 5;
    }
    void printAandB()
    {
        print();
        std::cout << "class b" << std::endl;
    }
};


int main(int argc, char** argv)
{
    std::cout << "My Module" << std::endl;
    int a = 0;
    CMath math_instance;
    a = math_instance.sum_number(1, 
                                 2);

    math_instance.print_out( a );

    CMath * math_pointerInstance = new CMath(1, 2);

    a = math_pointerInstance->sum_number( 1, 2 );
    math_pointerInstance->print_out( a );

    delete math_pointerInstance;
    math_pointerInstance = nullptr;

    //Adding Refence a in pointer variable c
    int * c = &a;
    std::cout << "Pointer varialbe: " << c << std::endl;
    std::cout << "address: " << &c << std::endl;
    std::cout << "de-referance: " << (*c) << std::endl;

    enum my_suit
    {
        Diamnds = 7,
        Hearts,
        Clubs,
        Spades
    };

    B instance_b;
    instance_b.printAandB();
    instance_b.print();

    return 0;
}