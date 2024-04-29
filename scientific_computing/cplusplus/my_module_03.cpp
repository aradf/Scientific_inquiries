#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <bitset>       // Fixed-size sequence of N bits.
#include <complex>      // Represent purely imaginary numbers.
#include <cstring>      // memset

/*
 * g++ -g -o ./run_compute my_module_03.cpp
 */

// get_max is a function template of class T type.  
// return class T and takes class T
namespace scientific_computing
{
    enum program_enum
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

    // Write a function to simulate 'endl'  
    // this type of function is called manpulator
    std::ostream& endl(std::ostream& some_stream)
    {
       some_stream.put('\n');
       some_stream.flush();
       return some_stream;
    }

    // ostream& ostream::operator<<(std::ostream& (*func)(ostream&))
    // return (*func)(*this);

    // '(*func)(ostream&)' is the pointer to a function
    // '<<' is an operator over load to take a pointer function as a parameter.
    // '(*func)(*this)' will invoke the '(*func)' function on the '(*this) object'

}


int main(int argc, char** argv)
{
    std::cout << "My Module" << std::endl;
    /*
     * C++ input/output library -- stream
     * cout:  a global object of ostream (typedef basic_ostream<char> ostream)
     * <<: ostream& ostream::operator << (string v);
     * endl: '\n' + flash
     * <<: ostream& ostre
     * What is stream: serial IO interface to external devices (file, stdin/stdout, network, etc..)
     */

    std::string s = "hello world";
    s[3] = 't';  //random access

    // name space scientific computing has a function template
    // with class T of long variable,
    long long_value = scientific_computing::get_max<long>(1, 2);
    
    std::cout << "scientific_computing::program_enum::first = " 
              << scientific_computing::program_enum::first 
              << std::endl;
    
    {
        // There is no need to open and/or close the ofstream, since it resides inside
        // a pair of curly brackets.  Entrying the curly bracket will open the ofstream
        // and leaving the curly bracket will close the ofstream.

        std::ofstream out_putFile("mylog.txt");
        out_putFile << "Experince is the mother of wisdom" << std::endl;
        out_putFile << 234 << " " << 2.3 << " " << std::endl; 
        out_putFile << std::bitset<8>(14) << std::endl;
        out_putFile << std::complex<int> (2,3) << std::endl;
    }   // typical Resource Acquisition is initialization technique: (RAII) 

    // IO operation: formatting the data and communicating the data with external devices.

    // Software Engineering Principal:  Low Coupling:  promots reusablity.
    {
        // There is no need to open and/or close the ofstream, since it resides inside
        // a pair of curly brackets.  Entrying the curly bracket will open the ofstream
        // and leaving the curly bracket will close the ofstream.

        // move the output pointer to the end of the file.
        // std is the name space of standard library
        // ofstream is the class template
        std::ofstream out_putFile("mylog.txt", std::ofstream::app);
        out_putFile << "Honesty is the best policy " << std::endl;
    }   

    {
        // the curly bracket removes the need to open and close the file.
        // the std is the standard template library.
        // ofstream is a class template
        std::ofstream out_putFile("mylog.txt", std::ofstream::in | std::ofstream::out);
        
        // move output pointer 10 chars (bytes) after begin position
        // standard library std is a namespace, 
        // over write five chars.
        out_putFile.seekp( 10, std::ios::beg);
        out_putFile << "12345" << std::endl;  
        
        // move output pointer 5 chars before end position
        out_putFile.seekp( -5, std::ios::end);
        out_putFile << "Nothing ventured, nothing gained." << std::endl;  

        // move output pointer 5 chars before the current position.
        out_putFile.seekp( -5, std::ios::cur);
    }   

    {
        std::ifstream input_file("mylog.txt");
        int i;
        input_file >> i;  /// read one word from the file and pass to i.
        // Error status: goodbit, badbit, failedbit, eofbit
        bool status = false;
        status = input_file.good();   //everything is o.k. goodbit == 1;
        status = input_file.bad();    //everything is o.k. badbit == 1;
        status = input_file.fail();   //everything is o.k. failbit == 1 and badbit == 1; 
        status = input_file.eof();    //everything is o.k. failbit == 1 and badbit == 1; 

        input_file.clear();           //clear all the error status
        // standard template library std, input output stream (ios)
        input_file.clear(std::ios::badbit);  // set a new value ot the error flag.
        input_file.rdstate();                // read the current status flag.
        input_file.clear(input_file.rdstate() & ~std::ios::failbit);  // clear only the failbit
        
        // Equivalent to: if (!input_file.fail())
        if (input_file)
            std::cout << "Read Succesfuly" 
                      << std::endl;
        if (input_file >> i)
            std::cout << "Read Succesfuly" 
                      << std::endl;
        // Handle the errors with exception
        // Setting the exception mask.
        try{
            input_file.exceptions(std::ios::badbit | std::ios::failbit);
        }
        catch ( ... )
            {
                std::cout << "Exception Thrown ... "
                          << std::endl;

            }
        // when badbit or failbit set to 1, exception of ios::failure will be thrown.
        // when eofbit set to 1, no exception will be thrown
        input_file.exceptions(std::ios::goodbit);
    }   

    {
        std::cout << 34 
                  << std::endl;
        
        // standard template library (stl), input output stream enumerator (ios)
        std::cout.setf(std::ios::oct, 
                       std::ios::basefield);
        
        // print out 42
        std::cout << 34
                  << std::endl;

        // print out 042
        std::cout.setf(std::ios::showbase);
        std::cout << 34
                  << std::endl;

        // print out 0x22
        std::cout.setf(std::ios::hex,
                       std::ios::basefield);
        std::cout << 34
                  << std::endl;

        // print out 22
        std::cout.unsetf(std::ios::showbase);
        std::cout << 34
                  << std::endl;

       // print out 34
        std::cout.setf(std::ios::dec,
                       std::ios::basefield);
        std::cout << 34
                  << std::endl;
    }

    {
        //          26
        std::cout.width(10);
        std::cout << 26 
                  << std::endl;
        std::cout.setf(std::ios::left, 
                       std::ios::adjustfield);
        std::cout << 26 
                  << std::endl;

        // Floating point values
        std::cout.setf(std::ios::scientific, 
                       std::ios::floatfield);

        // 3.400100E+02
        std::cout << 340.01 
                  << std::endl;

        // Floating point values
        std::cout.setf(std::ios::fixed, 
                       std::ios::floatfield);

        // 340.010000
        std::cout << 340.01 
                  << std::endl;

        // 340.010
        std::cout.precision(3);
        std::cout << 340.01 
                  << std::endl;

        /*
        int i;
        std::cin.setf(std::ios::hex, 
                      std::ios::basefield);
       
        std::cin >> i;           //Enter 12
        std::cout << i 
                  << std::endl;  //print 18
        */ 

        std::ios::fmtflags f = std::cout.flags();
        std::cout.flags(std::ios::oct | std::ios::showbase );
    }
    
    {
        // Member function for unformatted Input Output (IO)
        // input
        std::ifstream input_file("mylog.txt");
        const std::size_t buffer_length = 80;
        char buffer[buffer_length];
        std::memset(buffer, '\0', buffer_length);

        // Read up to 80 chars and save into buffer
        input_file.get(buffer, 
                       80);

        // Read up to 80 chars or unit '\n'
        input_file.getline(buffer, 
                           80);

        // Read up to 20
        input_file.read(buffer, 
                        20);

        input_file.ignore(3);

        // Returns the next character in the input sequence, without extracting it: 
        // The character is left as the next character to be extracted from the stream.
   
        char next_char = '\0';
        char current_char = '\0';

        next_char = input_file.peek();

        // return a char back to the stream
        // input_file.putback('z');
        input_file.unget();  

        // Extracts characters from the stream, as unformatted input:
        next_char = input_file.peek();
        current_char = input_file.get();
        next_char = input_file.peek();
        
        // return the number of chars being read by last unformtted read.
        current_char = input_file.get();
        int left_count = input_file.gcount();
    }

    {
        const std::size_t buffer_length = 80;
        char buffer[buffer_length];
        std::memset(buffer, '\0', buffer_length);
        std::memcpy(buffer, "hello world", 11);

        std::ofstream output_file("mylog.txt");
        output_file.put('c');
        output_file.put(' ');
        output_file.write(buffer, 11);
        output_file.flush();
    }   

    {
        std::ofstream output_file("mylog.txt");
        // endl: '\n' and flush
        // Is 'endl' an object, build-in data type, function?
        // <<: is the output operator
        std::cout << "hell world" 
                  << std::endl;

        scientific_computing::endl(output_file);
        std::cout << std::ends;  // insert a null char '\0'
        std::cout << std::flush; // flush the stream.

        // read and discard white space.
        // std::cin >> ws;       
    
        // 99________
        // std::cout << std::setw(8) 
        //           << std::left 
        //           << std::setfill('_')
        //           << 99 
        //           << std::endl;
          
        std::cout << std::hex 
                  << std::showbase 
                  << 14 
                  << std::endl;
    }

    return 0;
}