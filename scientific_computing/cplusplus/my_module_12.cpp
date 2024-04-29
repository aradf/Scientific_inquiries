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

/*
 * Unique Pointers Exlusive ownership, light weight smart pointer.
 */

class CDog
{
public:    
    std::string name_;

    CDog()
    {
        name_ = "nameless";
        std::cout << "CDog is created ..." << name_ << std::endl;
    }

    CDog(std::string name)
    {
        name_ = name;
        std::cout << "CDog is created ..." << name_ << std::endl;
    }

    ~CDog()
    {
        std::cout << "CDog is destroyed ..." << name_ << std::endl;
    }

    void bark()
    {
        std::cout << "CDog " << name_ << " barks ... " << std::endl;
    }

};


class CBone
{
public:    
   std::string name_;
   CBone(){}
   CBone(std::string name) {name_ = name;}
   ~CBone(){std::cout << "CBone " << name_ << "is deleted ..." << std::endl;}
   // copy assingment operator
   CBone& operator=(const CBone& other)
   {
       if (this != &other)
       {
           name_ = other.name_;
       }
       return *this;
   }


};

class BDog
{
    // declare a unique pointer of class type CBone.
    std::unique_ptr<CBone> pointer_bone;
public:    
    std::string name_;
    BDog()
    {
        name_ = "nameless";
        pointer_bone = std::make_unique<CBone>("hello");
        std::cout << "BDog is created ..." << name_ << std::endl;
    }

    BDog(std::string name)
    {
        name_ = name;
        std::cout << "BDog is created ..." << name_ << std::endl;
        pointer_bone = std::make_unique<CBone>("hello");
    }

    ~BDog()
    {
        std::cout << "BDog is destroyed ..." << name_ << std::endl;
    }

    void bark()
    {
        std::cout << "BDog " << name_ << " barks ... " << std::endl;
    }

};

// Scientific Computing Library
namespace scl
{
    // scientific Computing Enum
    void test()
    {
        // allocate memory on the heap.
        CDog * ptr_dog = new CDog("Gunner");
        ptr_dog->bark();
        // ptr_dog does a few things.
        // if there are some issues and the program counter leaves the scope 
        // of this function before the deleter is invoked, the code has memory leak.
        delete ptr_dog;
        ptr_dog = nullptr;
    }

    void test_uniquePtr()
    {
        std::unique_ptr<CDog> unique_ptrCDog (new CDog("Gunner"));
        unique_ptrCDog->bark();

        // The release method returns a pointer the CDog pointer object
        // It gives up the access to the pointer as well.
        CDog * ptr = unique_ptrCDog.release();

        if (!unique_ptrCDog)
        {
            std::cout << "unique_ptrCDog is Empty ..."  << std::endl;
            // CDog * ptr = unique_ptrCDog.release();
            delete ptr;
            ptr = nullptr;
        }
        
    }

    void test_uniquePtr2()
    {
        std::unique_ptr<CDog> unique_ptrCDog (new CDog("Gunner"));
        unique_ptrCDog->bark();

        // The reset method destroyes or deletes the original 'Gunner'
        unique_ptrCDog.reset(new CDog("Smokey"));

        if (!unique_ptrCDog)
        {
            std::cout << "unique_ptrCDog is Empty ..."  << std::endl;
        }
        else
        {
            std::cout << "unique_ptrCDog is not Empty ..."  << std::endl;
        }
        
    }

    void test_uniquePtr3()
    {
        std::unique_ptr<CDog> unique_ptrCDog (new CDog("Gunner"));
        unique_ptrCDog->bark();

        // The reset method destroyes or deletes the original 'Gunner'
        unique_ptrCDog.reset();   // The same as setting the unique_ptrCDog == nullptr;

        if (!unique_ptrCDog)
        {
            std::cout << "unique_ptrCDog is Empty ..."  << std::endl;
        }
        else
        {
            std::cout << "unique_ptrCDog is not Empty ..."  << std::endl;
        }
    }

    void test_uniquePtr4()
    {
        std::unique_ptr<CDog> unique_ptrCDog1 (new CDog("Gunner"));
        std::unique_ptr<CDog> unique_ptrCDog2 (new CDog("Smokey"));
        unique_ptrCDog1->bark();
        unique_ptrCDog2->bark();  
        //    move 
        // 1. Smokey is destroyed.
        // 2. unique_ptrCDog1 becomes empty.
        // 3. unique_ptrCDog2 owns gunner.
        unique_ptrCDog2 = std::move(unique_ptrCDog1);
        unique_ptrCDog2->bark();  
    }

    void do_something(std::unique_ptr<CDog> & some_pointer)
    {
        some_pointer->bark();
    }

    void test_uniquePtr5()
    {
        std::unique_ptr<CDog> unique_ptrCDog1 (new CDog("Gunner"));
        do_something(unique_ptrCDog1);
        if (!unique_ptrCDog1)
        {
            std::cout << "unique_ptrCDog1 is empty" << std::endl;
        }
    }

    std::unique_ptr<CDog> return_dog()
    {
        std::unique_ptr<CDog> p(new CDog("Smokey"));
        // automaticly the std::move semantic is invoked.
        // when returned, the p unique pointer no longer owns the 'smoky'.
        return p;
    }

    void test_uniquePtr6()
    {
        std::unique_ptr<CDog> unique_ptrCDog1 (new CDog("Gunner"));
        std::unique_ptr<CDog> unique_ptrCDog2 = return_dog();

        if (!unique_ptrCDog1)
        {
            std::cout << "unique_ptrCDog1 is empty" << std::endl;
        }

        if (!unique_ptrCDog2)
        {
            std::cout << "unique_ptrCDog2 is empty" << std::endl;
        }

        std::unique_ptr<CDog []> dogs(new CDog[3]);
        std::cout << std::endl;

        dogs.get()[0].name_ = "zero";
        dogs.get()[1].name_ = "one";
        dogs.get()[2].name_ = "two";                

        dogs.get()[0].bark();
        std::cout << std::endl;
    }

    void test_uniquePtr7()
    {
        std::unique_ptr<BDog> unique_ptrBDog (new BDog("Gunner"));

    }


}


int main(int argc, char** argv)
{
    std::cout << "Hello world" << std::endl;
    scl::test();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    scl::test_uniquePtr();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    scl::test_uniquePtr2();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    scl::test_uniquePtr3();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    scl::test_uniquePtr4();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    scl::test_uniquePtr5();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    scl::test_uniquePtr6();
   std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    scl::test_uniquePtr7();
      return 0;
}