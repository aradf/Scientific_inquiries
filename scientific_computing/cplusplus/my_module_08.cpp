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
 * g++ -std=c++17 -g -o ./run_compute my_module_08.cpp
 */

// Scientific Computing Library
namespace scl
{
    // scientific Computing Enum

}

// over load a function based on parameter type.

// int & i; means i is a l-value reference variable. l-value mean it is location in memory block 
// that can store an r-value or temperatry nominal value.
void print_int(int & i)
{
    std::cout << "l-value reference: " << i << std::endl;
}

// int && i; means i is a r-value reference that is referencing . l-value mean it is location in memory block 
// that can store an r-value or temperatry nominal value.
void print_int(int && i)
{
    std::cout << "r-value reference: " << i << std::endl;
}

class BVector
{
    int size;
    // a big array.  arr_ is a pointer variable. arr_ == 0x1234, *arr_ == 5, 
    // &arr_ == 0xABCD.  
    double * arr_;   
public:
    BVector()
    {
        size = 0;
        arr_ = nullptr;
    }
    // copy construct:  rhs is an l-value referance object.  l-value is a location in the memory block
    // where a r-value object can be stored.  r-value object is temporary and can not be used for 
    // purposes storage.  It is a nominal value.
    // copy constructor makes an expensive deep copy.
    BVector( const BVector& rhs)
    {
        size = rhs.size;
        arr_ = new double[size];
        for (int icnt = 0; icnt < size; icnt++)
        {
            arr_[icnt] = rhs.arr_[icnt];
        }
    }
    
    // Move constructor: rhs is an r-value referance that is refering.  l-value is a location in the 
    // memory block where r-value object can be stored.  r-value object is temporary and can not be
    // used for stroage.  It is a nominal value.
    // move constructor makes an in-expensive shallow copy.
    BVector( BVector&& rhs)
    {
        size = rhs.size;
        arr_ = rhs.arr_;
        rhs.arr_ = nullptr;
    }

    ~BVector() 
    { 
        delete arr_; 
        arr_ = nullptr;
    }
};

void foo(BVector vec)
{

}

template< typename T>
void relay(T&& arg)
{
    foo ( std::forward<T>( arg ) );
}

// BVector& vec means that vec is l-value.  It is a location in memory block where 
// r-value or content can be stored.
void foo_byReferance(const BVector& vec)
{

}

BVector create_bvector()
{
    BVector a_vector;
    return a_vector;
}

// pass by reference if you use arg to carry data back from goo to.
void goo(std::vector<int>& vec)
{
    vec.push_back(5);
}

// Ideally, alwayas attach units to the value;
// Usr defined literals quote quote and 'cm' is the identifier.  
// It multiplys by 10 and return long double.

constexpr long double operator""_cm(long double x) {return x * 10;}
constexpr long double operator""_m(long double x) {return x * 1000;}
constexpr long double operator""_mm(long double x) {return x;}

// Example to convert a string to binary.
int operator"" _bin(const char * some_string, size_t string_size)
{
    int return_binary = 0;
    for (int iCnt = 0; iCnt < string_size; iCnt++)
    {
        return_binary = return_binary << 1;
        if (some_string[iCnt] == '1')
        {
            return_binary += 1;
        }
    }
    return return_binary;
}

// Compiler error: overloaded function print_int is ambiguous.
// The compiler does not know which method to call.
// void print_int ( int i ) {}

int main(int argc, char** argv)
{
    std::cout << "Hello world" << std::endl;
    // C++ 11 Rvalue Referance
    // 1. Moving semantic
    // 2. Perfect Forwarding.
    {
       // a is l-vlaue refereance.  It is a location or address on the 
       // memorey block where a termperatry value like r-value 5 can 
       // be stored at.

       // b is a l-value referance variable.  That is a l-value that is
       // referancing a.  Before C++ 11 it was called (Reference)

       // c is r-value reference in other words c is an r-value that is referencing a referance. 

       int a = 5;
       int& b = a;
       int&& c = 5 ;   

       print_int (a);       // print_int(int & i)
       print_int (b);       // print_int(int & i)
       print_int (c);       // print_int(int & i)
       print_int (5);       // print_int(int && i)

       std::cout << a << " " << b << " " << c << std::endl;;
    }
    
    {
        BVector reusable = create_bvector();
        // expensive copy constructor, since reusable is l-value 
        foo(reusable);                // call copy constrctor.
        // reusable is destroyed.
        // reusable.arr_ == nullptr;
        // reusable is destroyed.
        foo(std::move(reusable));     // call the move constructor.

        // The create_bvector returns a r-value which is temporary and is destroyed 
        // momenterary.  It is not a location or memeroy address to store content.
        // in-expensive move constructor, since create_bvector() method returns r-value.
        foo_byReferance(create_bvector());
        foo_byReferance(reusable);   // will not call any constructor.
    }

    {
        // Move constructor / move assignment operator:
        // Purpose: conveniently avoid costly and unnessary keep copying.
        // they are particualrly powerful where passing by reference
        // and passing by value are both used.

        // they give you finer control of which part of your object to be moved.
 
        std::vector<int> my_vector = {1, 2, 3, 4};
        goo(my_vector);
        for (std::vector<int>::const_iterator vector_itr = my_vector.begin();
                                                vector_itr != my_vector.end();
                                                vector_itr++)
        {
            std::cout << *vector_itr << ' ';
        }
        std::cout << std::endl;

    }
 
    {
        // perfect forwarding ...
        BVector reusable = create_bvector();
        // reusable is l-value since is has a location in memory where temporary content like r-values 
        // can be storaged.  This function will invoke the copy constructor for deep copy.
        relay(reusable);

        // a r-value is passed on to relay, since does not have a location in memory where temporary 
        // content like r-values can be storaged.  It is a temporary content.  This function will invoke
        // the move constructor for shallow copy.
        relay(create_bvector());

        // 1. No costly and uncesary copy construction of BVector is made;  
        // 2. r-alue is forwarded as r-value, and l-value is forwarded as l-value.

        
    }
 
    {
        // Reference Collapsing Rules (c++ 11)
        // 1. T& &    ==> T&           //T reference to a referance.
        // 2. T& &&   ==> T&           //T reference to double reference.
        // 3. T&& &   ==> T&           //T double reference to reference 
        // 4. T&& &&  ==> T&&          //T double reference to double reference.

        // Remove reference on type T
        // template< class T>
        // struct remove_reference;

        // T is int&
        std::remove_reference<int&>::type i;        // int i;

        // T is int
        // std::remove_reference<int>::type i;         // compiler eror.

    }

    {
        // User define literals (actual or factual),
        // Lterals are constants.
        // c++ has 4 kinds of Literals
        // interger 
        // floating point
        // charter
        // string 
        int i = 45;
        unsigned ui = 45u;
        long     li = 45l;
        float    fi = 4.5f;
        // old c++ standard: 99
        long double height = 3.4f; //metrics, centimeters, inches.

        // Ideally, alwayas attach units to the value;
        // Usr defined literals quote quote and 'cm' is the identifier.  
        // It multiplys by 10 and return long double.  
        // the constexpr impact is to make the calculation at compile time, vs run time.

        // constexpr long double operator""_cm(long double x) {return x * 10;}
        // constexpr long double operator""_m(long double x) {return x * 1000;}
        // constexpr long double operator""_mm(long double x) {return x;}
        long double my_height = 3.4_cm;
        std::cout << my_height << std::endl;
        long double radius = 3.4_cm * 1.1_mm / 1.0_m;
        std::cout << radius << std::endl;
        std::cout << (my_height + 13.0_m) << " (cm) " << std::endl;
        std::cout << (130.0_mm / 13.0_m)  << std::endl;

        std::cout << "110"_bin << std::endl;
        std::cout << "1100110"_bin << std::endl;
        std::cout << "110100010001001110001"_bin << std::endl;

        std::cout << std::endl;
        // C++ went a long way to make suer defined types (classes) to behave same as build-in types.
        // User defined literals puashes this effort even further.

        // Restriction: It can only work on the following parameters.
        // char const*
        // unsigned long long
        // long double
        // char const*, std::size_t
        // wchar_t const*, std::size_t
        // char16_t const*, std::size_t
        // char32_t const*, std::size_t
        // Note the return value can be of any kind.

    }

    std::cout << std::endl;
    return 0;
}