/*
 * STL Headers
 */
#include <cstdlib>
#include <iostream>   // std::cin and std::cout
#include <fstream>    // std::ofstream
#include <thread>     // std::thread.
#include <mutex>      // std::mutex.
/*
 * g++ -std=c++17 -pthread -g -o ./run_compute my_module_37.cpp
 */

// Scientific Computing Library
namespace scl
{

// resource f is under the protection of the mutex.
class CLogFile
{
private:
   std::mutex my_mutex;
   std::ofstream f;
public:
   CLogFile()
   {
      // need destructor to close.
      f.open("log.txt");
   }
   ~CLogFile()
   {
      f.close();
   }
   // the data function takes stl class type string and integer.
   void shared_print(std::string id, int value)
   {
      // stl tempalte 'lock_gaurd' contains stl 'mutex'
      std::lock_guard<std::mutex> locker(my_mutex);
      f << "From " << id << ": " << value << std::endl;
   }
   /*
    * Never return f to the outside world.
    * // referance or address of the stl class type ofstream.
    * std::ofstream& get_stream() {return f;}
    */

   /*
    * Never pass f as an argument to a user provided function.
      void process(void fun(std::ofstream& ))
      {
         fun(f);
      }
    */
};

// Thread safe: Multiple threads can share a common resource 
// in a synchronized manner.
class CMyStack
{
private:
   int * data_;
   int i;
   std::mutex mu_;
public:
   // this function is not exception safe.
   int& pop()       // pops off the item on top of the stack.
   {
      return i;
   }     // returns the item on top.
};

void function_03(CMyStack& st)
{
   int v = st.pop();
   //process (v);
}

}  // end: namespace scl

//stl class type mutex
std::mutex my_mutex;

// thread function returning a void.  it takes on a
// stl class type string and int.
void shared_print(std::string msg, int id)
{
   // lock the std::mutex.
   // gaurantee on thread uses the resource at a time.
   // once the lock is in place.  if the instruction
   // my_mutex.lock() throws an exception, the 
   // code is in trouble.
   // my_mutex.lock();

   // stl template 'lock_guard' containing 'std::mutex'
   // initialized.  will gaurd agains exceptions.
   std::lock_guard<std::mutex> guard(my_mutex);


   std::cout << msg << " " << id << std::endl;
   // release the lock on the std::mutex.
   // my_mutex.unlock();
}


// thread function returning a void.  it takes on a
// stl class type string and int.
void shared_print_mutex_cout_are_not_bounded(std::string msg, int id)
{
   // lock the std::mutex.
   // gaurantee on thread uses the resource at a time.
   // once the lock is in place.  if the instruction
   // my_mutex.lock() throws an exception, the 
   // code is in trouble.
   // my_mutex.lock();

   // stl template 'lock_guard' containing 'std::mutex'
   // initialized.  will gaurd agains exceptions.
   std::lock_guard<std::mutex> guard(my_mutex);


   std::cout << msg << " " << id << std::endl;
   // release the lock on the std::mutex.
   // my_mutex.unlock();

}



void shared_print_has_issues_with_exceptions(std::string msg, int id)
{
   // lock the std::mutex.
   // gaurantee on thread uses the resource at a time.
   // once the lock is in place.  if the instruction
   // my_mutex.lock() throws an exception, the 
   // code is in trouble.
   my_mutex.lock();

   std::cout << msg << " " << id << std::endl;
   // release the lock on the std::mutex.
   my_mutex.unlock();
}

void function_01()
{
   for (int iCnt = 0; iCnt > -100; iCnt--)
   {
      // invoke the shared_print using stl class type string
      // and an integer.
      shared_print(std::string("From t1: "), iCnt);
   }
}

void function_02(scl::CLogFile& some_logfile)
{
   for (int iCnt = 0; iCnt > -100; iCnt--)
   {
      // invoke the shared_print using stl class type string
      // and an integer.
      some_logfile.shared_print(std::string("From t1: "), iCnt);
   }
}

/*
 * Both the working thread and main thread are racing for
 * the common resource the 'std::cout'.  The relative
 * execution order of either thread are not controlled.
 * std::mutex can be used to synchronized the acces of the 
 * threads.
 */

/*
 * Avoid Data Race
 * 1. Use std::mutex to syncrhonize data access (common resource).
 * 2. Never leak a handle of the data (common resource) to outside.
 * 3. Design Interface Approporately.
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   std::cout << "main thread id: " << std::this_thread::get_id() << std::endl;
   std::cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;

   {
      std::thread t1(function_01);
      std::cout << "t1 id: " << t1.get_id() << std::endl;

      try
      {
         for (int i=0; i< 100; i++)
         {
            std::cout << "main: "  << i << std::endl;
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
      std::thread t1(function_01);
      std::cout << "t1 id: " << t1.get_id() << std::endl;

      try
      {
         for (int i=0; i< 100; i++)
         {
            // invoke the shared_print method from main.
            // use the stl class type string and i
            shared_print(std::string("From main: "), i);
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
      scl::CLogFile log;

      // stl class type thread.  initialzied by thread function.
      // the log file is passed by refereance or address of the
      // log object.
      std::thread t1(function_02, std::ref(log));
      std::cout << "t1 id: " << t1.get_id() << std::endl;

      try
      {
         for (int i=0; i< 100; i++)
         {
            // invoke the shared_print method from main.
            // use the stl class type string and i
            log.shared_print(std::string("From main: "), i);
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


   std::cout << "Good Bye" << std::endl;
   return 0;
}