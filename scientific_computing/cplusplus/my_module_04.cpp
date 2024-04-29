#include <cstdlib>
#include <iostream>     // std::cin, std::cout
#include <string>       // std::string
#include <fstream>      // std::ofstream
#include <iterator>     // std::istreambuf_iterator
#include <sstream>      // std::stringstream

/*
 * g++ -g -o ./run_compute my_module_04.cpp
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

    template <typename T>
    void cout(T data)
    {
        std::cout << data << std::endl;
    }
    
    // Create Stream-enabled class
    struct Dog
    {
       int age_;
       std::string name_;
    };

    std::ostream& operator<<(std::ostream& sm, const Dog& d)
    {
        sm << "My name is "
           << d.name_ 
           << " and my age is "
           << d.age_
           << std::endl;
        
        return sm;
    }

    std::istream& operator>>(std::istream& sm, Dog& d)
    {
        sm >> d.age_;
        sm >> d.name_;
        return sm;
    }
}

int main(int argc, char** argv)
{
    std::cout << "Hello world" << std::endl;
    int i = 1;
    std::string str = "Scientfic Computing Library";
    scl::cout<int>(i);
    scl::cout<std::string>(str);
 
    // IO operation
    // Formatting data -- stream
    // Communicating data to external devices -- stream buffer
    std::cout << 34 << std::endl;
    // 'std::cout.rdbuf()' return a pointer object to the stream buffer.
    std::streambuf *ptr_streamBuffer = std::cout.rdbuf();

    {
        // the scientific_computingCout which is my cout is now pointing to the 
        // stream buffer that is internal to the std::cout 
        std::ostream scientific_computingCout(ptr_streamBuffer);
        scientific_computingCout << 34 << std::endl;
    
        scientific_computingCout.setf(std::ios::showpos);
        scientific_computingCout.width(20);
        //             +12
        scientific_computingCout << 12 << std::endl;
        //12
        std::cout << 12 << std::endl;

        std::ofstream output_file("mylog.txt");
        std::streambuf * orig_buf = std::cout.rdbuf();

        std::cout.rdbuf(output_file.rdbuf());
        std::cout << "Hello World" << std::endl;  // mylog.txt has hello world
        
        // re-directing the standard out to a log file.
        std::cout.rdbuf(orig_buf);
        std::cout << "Good Bye" << std::endl;  
    }

    // Stream Buffer iterator
    // std::istreambuf_iterator<char> input(std::cin);
    // std::ostreambuf_iterator<char> output(std::cout);
    // while (*input != 'x')
    // {
    //     *output = *input;
    //     ++output;
    //     ++input;
    // }

    // std::copy(std::istreambuf_iterator<char>(std::cin), 
    //           std::istreambuf_iterator<char>(),
    //           std::ostreambuf_iterator<char>(std::cout));

    // String Stream is a stream without IO operation.
    std::stringstream ss;
    // It can read/write to a string and from a string (treat string like a file)

    // Dec: 89 Hex: 59 Oct: 131
    ss << " Dec: "
       << 89 
       << " Hex: " 
       << std::hex 
       << 89 
       << " Oct: " 
       << std::oct 
       << 89;

    std::cout << ss.str() 
              << std::endl; 

    int a = 0, b = 0, c = 0;
    std::string s1;

    // Formtted input works token by token, space, tabs, newlines
    // ss >> std::hex >> a;
    // a == 137
    // s1: "hex"
    // ss >> s1;

    scl::Dog my_dog{2, "Bob"};   // universal initilaztion from std::c++ 11

    std::cout << my_dog;

    // std::cin >> my_dog;
    // std::cout << my_dog;
      

    return 0;
}