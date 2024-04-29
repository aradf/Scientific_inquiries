#include <cstdlib>
#include <iostream>     // std::cin, std::cout
#include <string>       // std::string
#include <fstream>      // std::ofstream
#include <iterator>     // std::istreambuf_iterator
#include <sstream>      // std::stringstream
#include <cstring>      // memset
#include <algorithm>   // count

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

    template <typename T>
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
    // Constructor
    std::string s1("hello");
    std::string s2("hello", 3);        // s2: hel
    std::string s3(s1);                // std::string copy constructor
    std::string s4(s1, 2);             // s4: llo
    std::string s5(s1, 2, 2);          // s5: ll
    std::string s6(5, 'a');            // s6: 'aaaaa'
    std::string s7({'a', 'b', 'c'});

    scl::cout<std::string>(s1);
    scl::cout<std::string>(s2);
    scl::cout<std::string>(s3);
    scl::cout<std::string>(s4);
    scl::cout<std::string>(s5);
    scl::cout<std::string>(s6);
    scl::cout<std::string>(s7);

    // Size or length of s1
    s1 = "GoodBye";
    int my_length = 0, my_capacity = 0;
    my_length = scl::size<std::string>(s1);
    my_capacity = s1.capacity();
    // size of the storage space currently allcoated to s1

    // allcoate 100 chars to the s1
    s1.reserve(100);
    std::cout << s1.capacity() << std::endl;

    // allcoate 5 chars to the s1: Goodbye s1.size() == 7 and s1.capacity() == 7
    s1.reserve(5);
    std::cout << s1.capacity() << std::endl;

    s1 = "GoodBye";
    s1.shrink_to_fit();
    std::cout << s1.length() << std::endl;
    std::cout << s1.capacity() << std::endl;

    // s1: GoodBye\0\0
    s1.resize(9);
    std::cout << s1.length() << std::endl;
    std::cout << s1.capacity() << std::endl;
    scl::cout<std::string>(s1);

    s1.resize(12, 'x');
    std::cout << s1.length() << std::endl;
    std::cout << s1.capacity() << std::endl;
    scl::cout<std::string>(s1);

    // s1: Goo
    s1.resize(3);
    std::cout << s1.length() << std::endl;
    std::cout << s1.capacity() << std::endl;
    scl::cout<std::string>(s1);

    {
       // Single Element Access
       std::string s1 = "GoodBye";
       std::cout << s1[2] << std::endl;      
       s1[2] = 'X';
       scl::cout<std::string>(s1);
       s1.at(2) = 'Y';
       scl::cout<std::string>(s1);

       // s1.at(20);  // Throw an exception of out_of_range
       // s1[20];     // Undefined behavior.
       
       std::cout << s1.front() << std::endl;      
       std::cout << s1.back() << std::endl;      
       s1.push_back('z');
       std::cout << s1 << std::endl;
       s1.pop_back();      
       std::cout << s1 << std::endl;

       for (std::string::iterator it=s1.begin(); 
            it !=s1.end(); 
            ++it)
       {
           std::cout << *it << " ";
       }
       std::cout << std::endl;
    }

    {
       std::string s1 = "GoodBye";       
       std::string s3(s1.begin(), s1.begin()+3);
       std::cout << s3 << std::endl;
    }

    {
        // Range Access of string: assing, append, insert, replace
        std::string s1;
        std::string s2 = "Dragon Land";
        s1.assign(s2);     // s1 = s2
        std::cout << s1 << "-" << s2 << std::endl;
        s1.assign(s2, 2, 4);   // s1 == agon
        std::cout << s1 << "-" << s2 << std::endl;
        s1.assign("Wisdom");
        std::cout << s1 << "-" << s2 << std::endl;
        s1.assign("Wisdom", 3);
        std::cout << s1 << "-" << s2 << std::endl;
        // s1.assign(s2,3);  // will not compile
        s1.assign(3, 'x');   // s1 = 'xxx'
        std::cout << s1 << "-" << s2 << std::endl;
        s1.assign({'a', 'b', 'c'});   // s1 = 'abc'
        std::cout << s1 << "-" << s2 << std::endl;

        s1.append(" def");  // s1 = abc def
        s1.insert(2, "mountain", 4);  // s1 = abmountc def
        s1.replace(2, 5, s2, 3, 3);   // s1 = abgon def

        s1.erase(1, 4);   // s1 = a def
        s2.substr(2, 4);  // return agon
        s1 = "abc";
        std::cout << s1.size() << std::endl;        
        s1.c_str();   // return "abc\0"
        std::cout << s1.size() << std::endl;
        s1.data();    // "abc"
        std::cout << s1.size() << std::endl;        

        s1.swap(s2);  //
    }

    {
        // Member Function Algorithms: copy, find, compare
        std::string s1 = "abcedefg";
        char buf[20];
        std::memset(buf, '\0', 20);         // copy null to buf
        std::size_t len = s1.copy(buf, 3);  // buf: abc len ==3 not null terminated.
        std::cout << s1 << std::endl;        
        std::cout << buf << std::endl;  
        std::memset(buf, '\0', 20);         // copy null to buf
        len  = s1.copy(buf, 4, 2);          // buf: cdef len = 4
        std::cout << s1 << std::endl;        
        std::cout << buf << std::endl;  
        
        s1 = "If a job is worth doing, it is worth doing well";
        std::size_t found = 0;
        found = s1.find("doing");             // found = 18
        found = s1.find("doing", 20);         // found = 37
        found = s1.find("doing well", 0, 5);  // found = 17       
        found = s1.find_first_of("doing");    // found = 6
        found = s1.find_first_of("doing", 10);    // found = 6
        found = s1.find_first_of("doing", 10, 1);    // found = 6
        found = s1.find_last_of("doing");    // found = 39
        found = s1.find_first_not_of("doing");    // found = 0
        found = s1.find_last_not_of("doing");    // found = 44

        s1.compare(s2);   //return positive if s1 > s2 and negative if s1 < s2 or zero if equal
        if (s1 > s2) {}   // if s1.compare(s2) > 0
        s1.compare(3, 2, s2);   // start the comparison from postion 3 and compare 2 char.
        std::string ss = s1 + s2;  // concatconate.
    }

    {
        // non-member functions
        std::string s1 = "abcedefg";
        std::cout << s1 << std::endl;

        // cin >> s1;
        // std::getline(std::cin, s1);    // default delimeter of '\n'
        // getline(std::cin, s1, ';');    // delimeter isi ';'

        // convert a number to string.
        s1 = std::to_string(8);
        std::cout << s1 << std::endl;
        s1 = std::to_string(2.3E7);     // s1 = 23000000.0000
        std::cout << s1 << std::endl;
        s1 = std::to_string(0xA4);      // s1 = 164 
        std::cout << s1 << std::endl;
        s1 = std::to_string(034);       // s1 = 28
        std::cout << s1 << std::endl;

        // convert a strign into a number
        s1 = "190";
        int i = 0;
        i = stoi(s1);

        s1 = "190 Monkey";
        i = stoi(s1);

        std::size_t pos;
        i = stoi(s1, &pos);  // i = 190 and pos = 3 which is the index of the string where the number stopped.

        s1 = "a Monkey";
        try{
             i = stoi(s1, &pos); // exception of invalid argument
        }
        catch ( ... )
            {
                std::cout << "Exception Thrown ... "
                          << std::endl;

            }
        i = stoi(s1, &pos, 16); // will see s1 as a Hex number so 'a' i = 10;

        // stoi, stod, stof, etc...
        // stringstream
        // lexical_cast() from boost library.
        // string and algorithms <string>

        s1 = "Variety is the spice of life";
        int num = 0;
        num = std::count(s1.begin(), s1.end(), 'e');   // num == 4

        // using a lambda function.
        num = std::count_if(s1.begin(), 
                            s1.end(), 
                            [](char c) {return ( c <= 'e' && c >= 'a');});  // num == 6

        s1 = "Goodness is better than beauty .";
        // itr pointer object will point to the first s of 'ss' in the first 20 char.
        std::string::iterator itr = std::search_n(s1.begin(), 
                                                  s1.end()+20, 
                                                  2, 
                                                  's');

        s1.erase(itr, itr+5);   // first five char will be remvoed from s1 at position of itr.
        std::cout << s1 << std::endl;

        s1.insert(itr, 3, 'x');
        std::cout << s1 << std::endl;

        s1.replace(itr, itr+3, 3, 'y');
        std::cout << s1 << std::endl;

        std::is_permutation( s1.begin(), 
                             s1.end(), 
                             s2.begin());

        std::replace( s1.begin(), s1.end(), 'e', ' ');
        std::cout << s1 << std::endl;

        // lambda function
        std::transform(s1.begin(), 
                       s1.end(), 
                       s2.begin(), 
                       [](char c) { if (c <'n')
                                        return 'a';
                                    else 
                                        return 'z';});

        std::cout << s1 << std::endl;
        std::cout << s2 << std::endl;

        s1 = "abcdefg";
        std::rotate(s1.begin(), s1.begin()+3, s1.end());
        std::cout << s1 << std::endl;

    }
    {
        std::string s;
        std::u16string s9; // string of 16 bit character char16_t type 
        std::u32string s8; // string of 32 bit character char16_t type 
        std::wstring s7;   // string wchar_t (wide character)

        std::cout << "what is up" << std::endl;

    }

    return 0;
}