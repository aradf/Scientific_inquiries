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

/*
 * g++ -g -o ./run_compute my_module_05.cpp
 */

// Scientific Computing Library
namespace scl
{
    // scientific Computing Enum
    enum sce
    {
        first,
        second,
        third
    };

    template <class T>
    T get_max(T a, T b)
    {
        T result;
        result = ( a > b ) ? a : b;
        return result;
    }

    template <class T>
    void cout(T data)
    {
        std::cout << data << std::endl;
    }
    
    template <typename T>
    int size(T data)
    {
        int object_length = data.length();
        int object_size = data.size();
        std::cout << data 
                  << " is " 
                  << object_length 
                  << " long. "
                  << std::endl;
        return (object_length);
    }
}

int main(int argc, char** argv)
{
    std::cout << "Hello world" << std::endl;
    // Learn C++ 11
    // Initialzier List
    int arr[4] = {3, 2, 4, 5};

    // Initializer List Constructor.
    std::vector<int> my_vector = {3, 2, 4, 5};
    // my_vector.push_back(3);
    // my_vector.push_back(2);
    // my_vector.push_back(4);
    // my_vector.push_back(5);

    char my_char = 'a';
    scl::cout<char >(my_char);

    for (std::vector<int>::const_iterator vector_itr = my_vector.begin();
                                                vector_itr != my_vector.end();
                                                vector_itr++)
    {
        std::cout << *vector_itr << ' ';
    }
    std::cout << std::endl;

    for (auto i = my_vector.begin();
                  i != my_vector.end();
                  i++)
    {
        std::cout << *i << ' ';
    }

    std::cout << std::endl;

    {
        // Define your own Initialize List Constructor.
        class CVector 
        {
           std::vector<int> my_vector;
        public:
           CVector(const std::initializer_list<int>& some_vector)
           {
             for (std::initializer_list<int>::iterator itr= some_vector.begin();
                  itr != some_vector.end();
                  itr++)
                  {
                    my_vector.push_back(*itr);
                  }
           }
           void cout()
           {
                for (std::vector<int>::const_iterator vector_itr = my_vector.begin();
                                                      vector_itr != my_vector.end();
                                                      vector_itr++)
                {
                    std::cout << *vector_itr << ' ';
                }
                std::cout << std::endl;
           
           }
        };

        try
        {
            CVector first_vector{0, 2, 3, 4, 5};
            first_vector.cout(); 

            CVector second_vector = {0, 2, 3, 4, 5};
            second_vector.cout(); 
        }
        catch ( ... )
            {
                std::cout << "Exception Thrown ... "
                          << std::endl;
            }
    }

    {
        // Uniform Initialization
        class dog 
        {
        public:
           int age;
           std::string name;
           void cout ()
           {
              std::cout << this->name 
                        << " - " 
                        << this->age 
                        << std::endl;

           }
        };
        
        // aggregate initializtion.
        dog dog1 = {5, "Henry"}; 
        dog1.cout();
    }

    {
        // Uniform Initialization
        class dog 
        {
        public:
           int age_;
           std::string name_;
           dog (int age, std::string name)
           {  
              this->age_ = age;
              this->name_ = name; 
           }
           void cout ()
           {
              std::cout << this->name_
                        << " - " 
                        << this->age_ 
                        << std::endl;

           }
       };
        
        // aggregate initializtion.
        dog dog1 = {6, "Half"}; 
        dog1.cout();
    }

    {
        /*
        * Uniform Initialization Search Order:
        * 1. Initializer_list constructor
        * 2. Regular constructor that takes the appropriete parameters
        * 3. Aggregate initializer
        */

        class dog 
        {
        public:
           // third choice.
           int age_;
            
           // second choice
           dog (int age)
           {  
              this->age_ = age;
           }
           
           // first choice
           dog (const std::initializer_list<int> & vec)
           {  
              this->age_ = *(vec.begin());
           }


       };
        
        // aggregate initializtion.
        dog dog1 = {6}; 
    }

    {
        std::vector<int> vec = {2,3,4,5};
        std::vector<int> m_vec;

        // c++ 03
        for (std::vector<int>::iterator iter = vec.begin();
                                        iter != vec.end();
                                        iter++)
        {
            m_vec.push_back(*iter);
        }                                        

        // c++ 11
        for (auto iter = vec.begin();
                  iter != vec.end();
                  iter++)
        {
            m_vec.push_back(*iter);
        }                                        

        auto a = 6;    // a is an integer
        auto b = 5.6;  // b is a float
        auto c = a;    // c is an integer. 

        for (auto i: vec)
        {
            std::cout << i;
        }

        for (auto& i: vec)
        {
            i++;
        }

    }


    return 0;
}