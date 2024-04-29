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

class CDog
{
    std::string name_;
public:    
    CDog(std::string name)
    {
        name_ = name;
        std::cout << "CDog is created ..." << name_ << std::endl;
    }
    CDog()
    {
        name_ = "nameless";
        std::cout << "CDog is created ..." << name_ << std::endl;
    }
    ~CDog () 
    {
        std::cout << "CDog is destroyed ..." << name_ << std::endl;
    }
    /*
     * Copy Constructor:
     * The other is an l-value reference object.  l-value is a location in the memory block.
     * where r-value or a temporary object can be stored.
     */
    CDog(const CDog& other)
    {
        name_ = other.name_;
    }
   /*
    * Copy Assignment operator:
    * The other is an l-value reference object.  l-value is a location in the memory block.
    * where r-value or a temporary object can be stored.
    */
    CDog& operator=(const CDog& other)
    {
        if (this != &other)
        {
            name_ = other.name_;
        }
        return *this;
    }

    void bark()
    {
        std::cout << "CDog " << name_ << " rules ... " << std::endl;
    }
};

// Scientific Computing Library
namespace scl
{
    // scientific Computing Enum
    void do_something()
    {
        // the ptr_someCDog is a pointer object with content 0x1234.  an address to another location 
        // in the memory block with the contenet of the CDog object.  *ptr_someCDog is the r-value 
        // content. 
        CDog * ptr_someCDog = new CDog("Gunner");
        
        // ...
        // if the ptr_someCDog is not deleted, there is memory leak.
        delete ptr_someCDog;
        // ...
     
        // Handle the errors with exception
        try
        {
            // ptr_someCDog is a dangling pointer - undefined behavior.
            ptr_someCDog->bark();
        }
        catch ( ... )
            {
                std::cout << "Exception Thrown ... "
                          << std::endl;
            }
    }

    void do_somethingSharedPointer()
    {
        // The class type is CDog, the name of the shared pointer object is sharedptr_someCDog
        // sharedptr_someCDog is l-value with 0x1234, the address of l-value is 0xABCD, 
        // *sharedptr_someCDog is an r-value CDog object.  It keeps track of how many pointers
        // are pointing to this pointer object location.  When the number becomes zero, the
        // sharedptr_someCDog will be deleted.  The sharedptr_someCDog's count is 1
        std::shared_ptr<CDog> sharedptr_someCDog(new CDog("Gunner"));
        std::cout << sharedptr_someCDog.use_count() << std::endl;

        std::cout << "Pause a Moment." << std::endl;    
        {
            std::shared_ptr<CDog> sharedptr_secondCDog = sharedptr_someCDog;
            std::cout << sharedptr_someCDog.use_count() << std::endl;
            std::cout << sharedptr_secondCDog.use_count() << std::endl;            
            
            // The sharedptr_someCDog's count is 2.
            // The sharedptr_secondCDog's count is 1.
            sharedptr_secondCDog->bark();
        }  // sharedptr_someCDog's count = 1
        std::cout << sharedptr_someCDog.use_count() << std::endl;

        // Pleaes note the '.' operator is used to access the shared memory pointer.
        // Please note the '->' opertor is used to access the class object.
        sharedptr_someCDog->bark();
    }   // sharedptr_someCDog's count = 0;

    void foo()
    {
        std::shared_ptr<CDog> sharedptr_firstCDog = std::make_shared<CDog>("Gunner");
        std::shared_ptr<CDog> sharedptr_secondCDog = std::make_shared<CDog>("Tank");    
        std::cout << sharedptr_firstCDog.use_count() << std::endl;
        std::cout << sharedptr_secondCDog.use_count() << std::endl;  
        sharedptr_firstCDog = sharedptr_secondCDog;  // "Gunner" is deleted.
        std::cout << sharedptr_firstCDog.use_count() << std::endl;
        std::cout << sharedptr_secondCDog.use_count() << std::endl;  

        sharedptr_firstCDog = nullptr;
        std::cout << sharedptr_firstCDog.use_count() << std::endl;
        std::cout << sharedptr_secondCDog.use_count() << std::endl;  
        sharedptr_firstCDog.reset();
        std::cout << sharedptr_firstCDog.use_count() << std::endl;
        std::cout << sharedptr_secondCDog.use_count() << std::endl;  
    }   

    void goo()
    {
          std::shared_ptr<CDog> p1 = std::make_shared<CDog>("Gunner");  // using default deleter: operator delete.
          // Using the lamda function to define custom deleter.
          std::shared_ptr<CDog> p2 = std::shared_ptr<CDog>(new CDog("Tank"),
                                     [](CDog * p)
                                     {
                                        std::cout << "Custome deleting ..."; delete p;
                                     }); 
        
          // create an array of dogs.  The p3 pointer points to the first element of 
          // the array.  When p3 goes out of scope, it only frees the memory for the first
          // element of the array.  The next two dogs have memory leak.
         
          /*
          std::shared_ptr<CDog> p3(new CDog[3]);    //<==== has memory leak (bad code.)
          */

          // the array delete[] will cause all three dogs to be free.  
          // when p4 goes out of scope.  No memory leaks.
          std::shared_ptr<CDog> p4(new CDog[3], 
                                   [](CDog * p) 
                                   { 
                                      std::cout << "Custome deleting ..."; delete[] p;
                                   });


    }


    /*
     Avoid using the raw pointer with the shared pointer.
     Avoid using the raw pointer with the shared pointer.
     Avoid using the raw pointer with the shared pointer.          

     shared pointer container is a way for a pointer to be shared.  when all of the objects of 
     the container go out of scope, the pointer will be deleted.
     */
    void boo()
    {
        // The CDog in the braket is the T.  The variable "gunner" is passed on.
        std::shared_ptr<CDog> p1 = std::make_shared<CDog>("Gunner");
        std::cout << p1.use_count() << std::endl;

        /*
        // The get method of the shared pointer will return a raw pointer of the object
        // in the container.
        CDog * raw_ptr = p1.get();

        // The raw_ptr is deleted. When the shared pointer goes out of scope, the raw_ptr is
        // deleted again which is undefined behavior.
        delete raw_ptr;
        std::cout << p1.use_count() << std::endl;
        std::cout << std::endl;
        
        // if a function takes raw_ptr as an input,  The second the function goes out of 
        // scope the raw_ptr could be deleted.
        doghouse.save(raw_ptr);

        */
    }

}

int main(int argc, char** argv)
{
    std::cout << "Hello world" << std::endl;
    // scl::do_something();

    scl::do_somethingSharedPointer();
        
    // Example of a bad way to use shared pointers.  An object should be assigned to a 
    // shared pointer right away.   Use the '.' or '->' arrow to access shared pointer
    // object ot the pointer object.
    CDog * some_cdog = new CDog("Tank");
    std::shared_ptr<CDog> p(some_cdog);       // p.use_count == 1
    std::cout << p.use_count() << std::endl;
    std::shared_ptr<CDog> p2(some_cdog);      // p2.use_count == 1
    std::cout << p2.use_count() << std::endl;
    std::cout << p.use_count() << std::endl;

    delete some_cdog;
    std::cout << p2.use_count() << std::endl;
    std::cout << p.use_count() << std::endl;

    // when p goes out of scope some_cdog pointer object is destroyed.
    // when p2 goes out of scope some_cdog pointer must be destroyed.

    // an object should be assigned to a smart pointer as soon as it is cretaed. 
    // Raw pointer should not be used again.  The proper way is below:
    std::shared_ptr<CDog> good_sharedPtr(new CDog("Gunner"));
    // 1. Raw Pointer of 'Gunner' is created.
    // 2. shared pointer of good_sharedPtr is created.

    // The prefared way of created a shared pointer is
    std::shared_ptr<CDog> prefered_sharedPtr = std::make_shared<CDog>("Tank");
    prefered_sharedPtr->bark();
    (*prefered_sharedPtr).bark();
    std::cout << prefered_sharedPtr.use_count() << std::endl;

    // static_pointer_cast
    // dynamic_pointer_cast
    // const_pointer_cast

    scl::foo();
    scl::goo();
    scl::boo();
    
    std::cout << std::endl;
    return 0;
}