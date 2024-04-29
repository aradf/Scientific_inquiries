/*
 * STL Headers
 */
#include <cstdlib>
#include <iostream>   // std::cin and std::cout
#include <fstream>    // std::ofstream
#include <thread>     // std::thread.
#include <mutex>      // std::mutex.
/*
 * g++ -std=c++17 -pthread -g -o ./run_compute my_module_38.cpp
 */

// Scientific Computing Library
namespace scl
{

// resource f is under the protection of the mutex.
class CLogFile
{
private:
   std::mutex my_mutex;
   std::mutex my_mutex2;
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
      std::lock(my_mutex, my_mutex2);
      std::lock_guard<std::mutex> locker(my_mutex, std::adopt_lock);
      std::lock_guard<std::mutex> locker2(my_mutex2, std::adopt_lock);
      // f << "From " << id << ": " << value << std::endl;
      std::cout << "From " << id << ": " << value << std::endl;

      /*
      {
         std::lock_guard<std::mutex> locker(my_mutex);
         // f << "From " << id << ": " << value << std::endl;
         std::cout << "From " << id << ": " << value << std::endl;
      }
      {
         std::lock_guard<std::mutex> locker2(my_mutex2, std::adopt_lock);
         /// .... 
      }
      */

   }
   void shared_print2(std::string id, int value)
   {
      // stl tempalte 'lock_gaurd' contains stl 'mutex'

      std::lock(my_mutex, my_mutex2);
      std::lock_guard<std::mutex> locker(my_mutex, std::adopt_lock);
      std::lock_guard<std::mutex> locker2(my_mutex2, std::adopt_lock);
      // f << "From " << id << ": " << value << std::endl;
      std::cout << "From " << id << ": " << value << std::endl;

      /*
      {
         std::lock_guard<std::mutex> locker(my_mutex);
         // ....
      }
      {
         std::lock_guard<std::mutex> locker2(my_mutex2, std::adopt_lock);
         // f << "From " << id << ": " << value << std::endl;
         std::cout << "From " << id << ": " << value << std::endl;
      }
      */
   }

};

void function_01(scl::CLogFile& some_logfile)
{
   for (int iCnt = 0; iCnt > -100; iCnt--)
   {
      // invoke the shared_print using stl class type string
      // and an integer.
      some_logfile.shared_print(std::string("t1: "), iCnt);
   }
}

}  // end: namespace scl

/*
 * Both the function_01 and main thread are printing to the std::cout in 
 * a syncronized manner.
 */

/*
 * classical deadlock situation.  the function shared_print locks my_mutex
 * and before it locks my_mutex2, the function shared_print2 locks my_mutex2.
 * The function shared_print is waiting on shared_print2 to release my_mutex2.
 * The function shared_print2 is waiting on shared_print to release my_mutex.
 * In a deadlock such this, two threads are wait on the each other the 
 * release the access to a shared resource.
 * 1. To avoid dead locks, make sure that both threads lock their mutexs in
 *    the same order.
 * 2. STL provides the std::lock data type to prevent this dead lock.
 * 3. Evaluate the situation, do you really need two mutexs.
 * 
 * Avoiding dead lock.
 * 1. Prefer locking single mutex.
 * 2. Avoid locking a mutex and then calling a user provided function.
 *    the user provided function could lock a mutex in the back ground.
 * 3. If need two mutex, use the std::lock() to lock more than one mutex.
 * 4. Lock the mutex in same order for all threads.
 * 
 * Locking Granularity:
 * - Fine-granied lock:  prtotects amll amount of data.
 * - Coarse-grained lock: protects big amount of data.
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   std::cout << "main thread id: " << std::this_thread::get_id() << std::endl;
   std::cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;
   {
      scl::CLogFile log;

      // stl class type thread.  initialzied by thread function.
      // the log file is passed by refereance or address of the
      // log object.
      std::thread t1(scl::function_01, std::ref(log));
      std::cout << "t1 id: " << t1.get_id() << std::endl;

      try
      {
         for (int i=0; i< 100; i++)
         {
            // invoke the shared_print method from main.
            // use the stl class type string and i
            log.shared_print2(std::string("main: "), i);
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