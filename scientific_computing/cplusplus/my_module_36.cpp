/*
 * STL Headers
 */
#include <cstdlib>
#include <iostream>   // std::cin and std::cout
#include <thread>     // std::thread.
/*
 * g++ -std=c++17 -pthread -g -o ./run_compute my_module_36.cpp
 */

// Scientific Computing Library
namespace scl
{

}  // end: namespace scl

void function_01()
{
   std::cout << "Beauty is only skin deep" << std::endl;
}

class CFunctor
{
public:
   void operator()()
   {
      for (int iCnt=0; iCnt>-100; iCnt--)
      {
         std::cout << "Cfunctor: " << iCnt << std::endl;
      }
   }

   void operator()(std::string msg)
   {
         std::cout << "Cfunctor: " << msg << std::endl;
   }
};

class CFunctor1
{
public:
   void operator()(std::string& msg)
   {
         std::cout << "Cfunctor: " << msg << std::endl;
         msg = "Trust is the mother of deceit.";
   }
};

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;

   std::cout << "main thread id: " << std::this_thread::get_id() << std::endl;

   {
      // t1 starts running.
      // stl class type thread.  Initialized with thread function 'function_01'
      std::thread t1(function_01);
      std::cout << "t1 thread id: " << t1.get_id() << std::endl;      
      // using RAII
      // Wrapper w(t1);

      try
      {
         for (int i=0; i< 100; i++)
         {
            std::cout << "from main: "  << i << std::endl;
         }
      }
      catch( ...)
      {
         // The main thread waits for t1 thread to finish.
         if (t1.joinable()) 
            t1.join(); 
         throw;
      }
      
      if (t1.joinable()) 
         t1.join(); 


      // t1 will freely on it's own. -- t1 becomes a daemon process.
      // t1.detach();
   }
  
   {
      CFunctor fct;
      //stl's class type thread:  initialized with fct.
      // start t1;
      std::thread t1(fct);  
      std::cout << "t1 thread id: " << t1.get_id() << std::endl;      
      try
      {
         for (int i=0; i< 100; i++)
         {
            std::cout << "from main: "  << i << std::endl;
         }
      }
      catch( ...)
      {
         // The main thread waits for t1 thread to finish.
         if (t1.joinable()) 
            t1.join(); 
         throw;
      }
      
      if (t1.joinable()) 
         t1.join(); 

   }

   {
      //stl's class type thread:  initialized with fct.
      // start t1;
      std::thread t1( ( CFunctor()) );  
      std::cout << "t1 thread id: " << t1.get_id() << std::endl;
      try
      {
         for (int i=0; i< 100; i++)
         {
            std::cout << "from main: "  << i << std::endl;
         }
      }
      catch( ...)
      {
         // The main thread waits for t1 thread to finish.
         if (t1.joinable()) 
            t1.join(); 
         throw;
      }
      
      if (t1.joinable()) 
         t1.join(); 

   }

  {
      std::string msg_string = "Where there is no trust, there is no love";
      //stl's class type thread:  initialized with fct.
      // start t1;
      std::thread t1( ( CFunctor()), msg_string);  
      std::cout << "t1 thread id: " << t1.get_id() << std::endl;

      try
      {
         std::cout << "main: "  << msg_string << std::endl;
      }
      catch( ...)
      {
         // The main thread waits for t1 thread to finish.
         if (t1.joinable()) 
            t1.join(); 
         throw;
      }
      
      if (t1.joinable()) 
         t1.join(); 

   }

  {
      std::string msg_string = "Where there is no trust, there is no love";
      //stl's class type thread:  initialized with fct.
      // start t1;
      std::thread t1( ( CFunctor1()), std::ref(msg_string));  
      std::cout << "t1 thread id: " << t1.get_id() << std::endl;

      if (t1.joinable()) 
         t1.join(); 

      std::cout << "main: "  << msg_string << std::endl;
   }

   {
      // oversubscription

      // stl class type thread.
      // Indication on how many threads can run concurrently ini the application.
      std::cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;

   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}