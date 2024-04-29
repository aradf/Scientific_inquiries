/*
 * STL Headers
 */
#include <cstdlib>
#include <iostream>   // std::cin and std::cout
#include <fstream>    // std::ofstream
#include <thread>     // std::thread.
#include <mutex>      // std::mutex.
#include <condition_variable>  // std::conition_variable.
#include <future>     // future

/*
 * g++ -std=c++17 -pthread -g -o ./run_compute my_module_41.cpp
 */

// Scientific Computing Library
namespace scl
{
   std::mutex my_mutex;
   std::condition_variable condition_variable;

   // function factorial returns a void, takes on integer n, takes on address of an integer
   // or a referance to an integer.  Please note that the referance to x is a shared resource
   // between the working thread and the main thread.
   void factorial(int n, int& x)
   {
      int result = 1;
      // stl template unique_lock contains stl class type mutex.
      std::unique_lock<std::mutex> locker(my_mutex, std::defer_lock);

      for (int i = n ; i > 1; i--)
      {
         result *= i;
      }
      std::cout << "factorial: " << result << std::endl;

      // stl template lock_gaurd contains a stl class type mutex.
      locker.lock();
      x = result;
      locker.unlock();
   }

   int factorial1(int n)
   {
      int result = 1;
      for (int i = n ; i > 1; i--)
      {
         result *= i;
      }
      return result;
   }

   // factorial2 return an integer and takes stl template future contain int which
   // is passed on the function by referance or address.
   int factorial2(std::future<int>& some_future)
   {
      int result = 1;
      int n = some_future.get();
      for (int i = n ; i > 1; i--)
      {
         result *= i;
      }
      return result;
   }

}  // end: namespace scl

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   std::cout << "main thread id: " << std::this_thread::get_id() << std::endl;
   std::cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;

   {
      int x=0;
      // the stl class type thread is initialized with thread fucntion factorial,
      // it takes on integer 4 and stl function member referance.
      std::thread t1(scl::factorial, 4, std::ref(x));   
      t1.join();
   }

   {
      int x=0;
      // stl future template contains integer and wait for the facotiral thread
      // to complete.
      
      // factorial thread is created ...
      // std::future<int> my_future = std::async(scl::factorial1, 4);
      
      // anoterh thread is not created ...
      // std::future<int> my_future = std::async(std::launch::deferred, scl::factorial1, 4);

      // factorial thread is created ...
      std::future<int> my_future = std::async(std::launch::async, scl::factorial1, 4);

      x = my_future.get();
      std::cout << "main: " << x << std::endl;
   }

   {
      int x=0;

      // stl template promise contains integer;
      // stl template future contains int;
      std::promise<int> my_promise;
      std::future<int> my_future = my_promise.get_future();

      // factorial thread is created ...
      // stl template future contains integer, stl function async starts factorial3 thread because of
      // std::launch::async, stl referance method enables my_future to be passed by referance or address.
      std::future<int> next_future = std::async(std::launch::async, scl::factorial2, std::ref(my_future));

      // do something else ...
      std::this_thread::sleep_for(std::chrono::milliseconds(20));

      // the child thread get value 4.
      my_promise.set_value(4);

      std::this_thread::sleep_for(std::chrono::milliseconds(20));

      x = next_future.get();
      std::cout << "main - promise - from child: " << x << std::endl;
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}