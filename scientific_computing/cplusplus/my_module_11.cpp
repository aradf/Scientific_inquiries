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
    // Memory is not released.  The reason is cyclic reference.
    // std::shared_ptr<CDog> m_pFriend;

    // The shared pointer allows for multiple owner ship of the pointer.
    // The weak pointer allows access and not ownership of the pointer.
    // when and how the object is created and deleted is out of the 
    // code developer's control.  The code only has access.
    // A weak pointer is similar to a raw pointer.  
    // it provides a safer access to the pointer.
    std::weak_ptr<CDog> m_pFriend;
public:    
    std::string name_;

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
        std::cout << "CDog " << name_ << " rules ... " << std::endl;
    }

    void makeFriend(std::shared_ptr<CDog> f)
    {
        this->m_pFriend = f;
    }

    void show_friend()
    {
        // The lock function creates a shared pointer from the weak pointer.
        if (!m_pFriend.expired())
            std::cout << "My friend is " << m_pFriend.lock()->name_ << std::endl;
        std::cout << "He is owned by " << m_pFriend.use_count() << " pointers." << std::endl;
    }
};

// Scientific Computing Library
namespace scl
{
    // scientific Computing Enum

}

/*
   Memory is not released.  The reason is cyclic reference.
 */

int main(int argc, char** argv)
{
    std::cout << "Hello world" << std::endl;
    // scl::do_something();
    std::shared_ptr<CDog> pD1(new CDog("Gunner"));
    std::shared_ptr<CDog> pD2(new CDog("Smokey"));
    std::cout << pD1.use_count() << std::endl;
    std::cout << pD2.use_count() << std::endl;
    pD1->makeFriend(pD2);
    std::cout << pD1.use_count() << std::endl;
    std::cout << pD2.use_count() << std::endl;
    pD2->makeFriend(pD1);
    std::cout << pD1.use_count() << std::endl;
    std::cout << pD2.use_count() << std::endl;
    pD1->show_friend();

    std::cout << std::endl;
    return 0;
}