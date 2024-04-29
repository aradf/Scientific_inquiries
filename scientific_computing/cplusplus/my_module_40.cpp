/*
 * STL Headers
 */
#include <cstdlib>
#include <iostream>   // std::cin and std::cout
#include <fstream>    // std::ofstream
#include <thread>     // std::thread.
#include <mutex>      // std::mutex.
#include <deque>      // std::deque
#include <chrono>     // std::chrono
#include <condition_variable>  // std::conditional_variable

/*
 * g++ -std=c++17 -pthread -g -o ./run_compute my_module_39.cpp
 */

// Scientific Computing Library
namespace scl
{

// stl class type mutex.
std::mutex my_mutex;

// stl template deque contain intger.
std::deque<int> my_deque;

// stl class type conditional_variable
std::condition_variable conditional_variable;

void function_produceData1()
{
   int count = 10;
   while ( count > 0 )
   {
      // stl unique_locker template containing mutex initialized.
      std::unique_lock<std::mutex> locker(my_mutex);
      my_deque.push_front(count);
      locker.unlock();

      // Notify one waitign thread, if there is one.
      conditional_variable.notify_one();

      // conditional_variable.notify_all();

      // stl this_thread::sleep_for function member chronos::seconds function
      // std::this_thread::sleep_for(std::chrono::seconds(1));
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      count--;
   }
}

void function_consumeData1()
{
   int data = 0;
   while (data != 1)
   {
      // stl template containing stl class type mutex.
      std::unique_lock<std::mutex> locker(my_mutex);

      // Will wait till the conditional variable notifies.
      // spurious wake.
      conditional_variable.wait(locker, [](){return !my_deque.empty();});
      data = my_deque.back();
      my_deque.pop_back();
      locker.unlock();
      std::cout << "t2 got a value from t1: " << data << std::endl;
   }
}

void function_produceData0()
{
   int count = 10;
   while ( count > 0 )
   {
      // stl unique_locker template containing mutex initialized.
      std::unique_lock<std::mutex> locker(my_mutex);
      my_deque.push_front(count);
      locker.unlock();

      // Notify one waitign thread, if there is one.
      conditional_variable.notify_one();

      // stl this_thread::sleep_for function member chronos::seconds function
      std::this_thread::sleep_for(std::chrono::seconds(1));
      count--;
   }
}

void function_consumeData0()
{
   int data = 0;
   while (data != 1)
   {
      // stl template containing stl class type mutex.
      std::unique_lock<std::mutex> locker(my_mutex);
      if (!my_deque.empty())
      {
         data = my_deque.back();
         my_deque.pop_back();
         locker.unlock();
         std::cout << "t2 got a value from t1: " << data << std::endl;
      }
      else
         {
            locker.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
         }
   }
}

}  // end: namespace scl

// Using std::mutex to synchronize the access of common resources among threads.
// my_daque is a shared resource or shared memory, the mutex will synchronize the
// access the my_daque between the producer thread and consumer thread.

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   std::cout << "main thread id: " << std::this_thread::get_id() << std::endl;
   std::cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;
 
   {
      std::thread t1(scl::function_produceData1);
      std::thread t2(scl::function_consumeData1);
      t1.join();
      t2.join();
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}