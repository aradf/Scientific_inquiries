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
 * g++ -g -o ./run_compute my_module_07.cpp
 */

// Scientific Computing Library
namespace scl
{
    // scientific Computing Enum
    void foo(int iValue)
    {
        std::cout << "foo_int" << std::endl;
    }

    // scientific Computing Enum
    void foo(char * ptr_char)
    {
        std::cout << "ptr_char" << std::endl;
    }

    // c++ 03
    enum apple 
    {
        green_apple, 
        red_apple
    };
    
    enum orange 
    {
        green_orange, 
        red_orange
    };

    // c++ 11
    enum class CApple
    {
        green, 
        red
    };

    enum class COrange {green, red};

}

namespace xyz
{

    template<typename func>
    void filter(func f, std::vector<int> arr)
    {
    for (auto i: arr)
    {
        if (f(i))
            std::cout << i << " ";
    }
    }
}


int main(int argc, char** argv)
{
    std::cout << "Hello world" << std::endl;
    /*
     * nullptr replaced NULL in c++ 03
     */

    // scl::foo(NULL); Would not compile due foo(NULL) is ambiguous
    scl::foo(nullptr);
    scl::apple my_apple = scl::green_apple;
    scl::orange my_orange = scl::green_orange;

    if (my_apple == scl::green_apple)
    {
        std::cout << "Green Apple" << std::endl;
    }

    scl::CApple some_apple = scl::CApple::green;
    scl::COrange some_orange = scl::COrange::red;

    if (some_apple == scl::CApple::green)
    {
        std::cout << "scl::apple_class::green" << std::endl;
    }
    
    // static assert - Run-time assert
    // a is an integer of 5, with &a = 0x1234
    // my_ptr is an address of 0xABCD, *a is an integer like 5, and &a is an address 0x4321
    int a = 5; 
    int * my_ptr = NULL;
    // assert (my_ptr != NULL);  // Code assert here.

    my_ptr = &a;
    assert (my_ptr != NULL);
   
    // compile time assert (c++ 11)
    static_assert( sizeof(int) == 4);

    {
        // Delegating Constructor
        class CDog
        {
         public:
            CDog() { std::cout << "CDOG" << std::endl; }
            CDog(int a) { CDog(); std::cout << "CDOG" << a << std::endl; }
        };

        CDog my_dog;
        CDog your_dog(0);
    }

    {
        // c++ 03
        class CDog
        {
            void init() { std::cout << "CDOG-init" << std::endl; }
        public:
            CDog() { init(); }
            CDog(int a) { init(); std::cout << "CDOG-dosomething" << a << std::endl; }
        };
        CDog my_dog;
        CDog your_dog(0);
    }

    {
        // c++ 11
        class CDog
        {
            int age = 9;
        public:
            CDog() { init(); }
            CDog(int a) : CDog() { init(); std::cout << "CDOG-dosomething" << a << std::endl; }
            void init() { std::cout << "CDOG-init" << std::endl; }
        };
        std::cout << std::endl << std::endl;
        CDog my_dog;
        CDog your_dog(0);
    }

    {
        // c++ 03
        // override for vitual function.
        // Avoid inadvertantly create new function in dervied classes.
        class CDog
        {
            int a_;
            virtual void A(int a) {a_ = a;}
            virtual void B() const {}
        public:
            CDog() { init(); }
            void init() { std::cout << "CDOG-init" << std::endl; }
        };
        
        class CYellowDog : public CDog
        {
            float a_;
            virtual void A(float a) {a_ = a;};  //create a new function.
            virtual void B() {};       // create a new function.

        };

        std::cout << std::endl << std::endl;
        CDog my_dog;
        CYellowDog my_yelloDog;
    }

    {
        // c++ 11
        // Use 'override' for vitual function to specify a method in the derived class
        // is in fact intended to over ride the method from parent class.
        // Avoid inadvertantly create new function in dervied classes.
        class CDog
        {
            int a_;
            virtual void A(int a) {a_ = a;}
            virtual void B() const {}
            void c() {};
        public:
            CDog() { init(); }
            void init() { std::cout << "CDOG-init" << std::endl; }
        };
        
        class CYellowDog : public CDog
        {
            float a_;
            // virtual void A(float a) override {a_ = a;};  //Does not create a new function.
            // virtual void B() override {};       //Does not create a new function.
            // void c() override;
        };

        std::cout << std::endl << std::endl;
        CDog my_dog;
        CYellowDog my_yelloDog;
    }

    {
        // 'final' key word no class can be derived from CDog
        // 'final' key word no class can be override for virtual function and for class.
        class CDog final 
        {
            int a_;
            virtual void A(int a) {a_ = a;}
            virtual void B() const {}
            virtual void c() final {};
        public:
            CDog() { init(); }
            void init() { std::cout << "CDOG-init" << std::endl; }
        };
        
        // class CYellowDog : public CDog
        // {
        //     float a_;
        //     // virtual void A(float a) override {a_ = a;};  //Does not create a new function.
        //     // virtual void B() override {};       //Does not create a new function.
        //     void c() {};
        // };

        std::cout << std::endl << std::endl;
        CDog my_dog;
        // CYellowDog my_yelloDog;
    }

    {
        // Compiler generated defaut constructor.
        class CDog 
        {
            int a_;
        public:
            CDog(int a) { a_ = a; }
            CDog() = default;           // Forces the compiler to geneate default constructor.
            void init() { std::cout << "CDOG-init" << std::endl; }
        };

        std::cout << std::endl << std::endl;
        CDog my_dog;
    }

    {
        // delete a function.
        class CDog 
        {
            int a_;
        public:
            CDog(int a) { a_ = a; }
            void init() { std::cout << "CDOG-init" << std::endl; }
        };

        std::cout << std::endl << std::endl;
        CDog my_dog(2);
        CDog a_dog(3.0);          // 3.0 is converted from double to int.
        my_dog = a_dog;           // compiler generated assignment operator
    }

    {
        // delete a function.
        class CDog 
        {
            int a_;
        public:
            CDog(int a) { a_ = a; }
            CDog(double ) = delete;
            CDog& operator=(const CDog&) = delete;
            void init() { std::cout << "CDOG-init" << std::endl; }
        };

        std::cout << std::endl << std::endl;
        CDog my_dog(2);
        // CDog a_dog(3.0);          // Generate compiler error: the double constructor is deleted.
        // my_dog = a_dog;           // Generate compiler error: compiler generated assignment operator is deleted.
    }

    {
        // constexpr
        class CDummy 
        {
            public:
            int arr[6];
            int A() { return 3; };
            // int arr[A() + 3];  // Compiler error.

            // C++ 11
            // constexpr int A() { return 3; }  //force the computation to happen at compile time.

            // int arr[A()+3];                // create an array of size 6
            // write faster program with constexpr
            constexpr int cubed(int x) {return x * x * x;}
            void my_method() 
            { int y = cubed(1789); }  // computed at compile time.
        };
        
        CDummy hi_dummy;
        hi_dummy.my_method();
    }

    {
        // new string literals c++ 03
        char * a = (char *)"string";

        // c++ 11
        char * e = (char *)u8"string";          // define an UTF-8 string
        char16_t * b = (char16_t *)u"string";       // define an UTF-16 string
        char32_t * c = (char32_t *)U"string";       // define an UTF-32 string.
        // char *     d = R"string \\"

    }

    {
        // Lambda function.
        std::cout << [](int x, int y) {return x+y;} (3,4) << std::endl;
        auto f = [](int x, int y) { return x + y; };
        std::cout << f(3,4) << std::endl; 

        std::vector<int> my_vector = {1, 2, 3, 4, 5, 6};
        xyz::filter( [] (int x) {return (x>3);}, my_vector);
        xyz::filter( [] (int x) {return (x>2 && x < 5);}, my_vector);

        int y = 4;
        // Note:  [&] tells compiler thta we want variable capture.
        xyz::filter( [&] (int x) {return (x>y);}, my_vector);

    }

    return 0;
}