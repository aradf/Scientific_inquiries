/*
 * STL Headers
 */
#include <cstdlib>
#include <iostream>   // std::cin and std::cout
#include <fstream>    // std::ofstream
#include <thread>     // std::thread.
#include <future>     // std::package_task
#include <functional> // std::bind
#include <deque>      // std::deque

/*
 * g++ -std=c++17 -pthread -g -o ./run_compute my_module_41.cpp
 */

// Scientific Computing Library
namespace scl
{

int factorial( int n)   
{
   int result = 1;
   for (int i = n; i>1; i--)
   {
      result *= i;
   }

   std::cout << "Result is: " << result  << std::endl;
   return result;
}

}  // end: namespace scl

/*
 * Packaged Task.
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   std::cout << "main thread id: " << std::this_thread::get_id() << std::endl;
   std::cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;

   {
      /* Threads*/
      std::thread t1(scl::factorial, 6);

      std::this_thread::sleep_for(std::chrono::milliseconds(3));
      std::chrono::steady_clock::time_point time_point = std::chrono::steady_clock::now() + std::chrono::microseconds(4);
      std::this_thread::sleep_until(time_point);

      /* mutex */
      std::mutex mu;
      std::lock_guard < std::mutex > locker(mu);
      std::unique_lock < std::mutex > ulocker(mu);

      ulocker.try_lock();
      // ulocker.try_lock_for(std::chrono::nanoseconds(500));
      // ulocker.try_lock_until(time_point);

      /* conditional variable */
      std::condition_variable conditional_variable;
      conditional_variable.wait_for(ulocker, std::chrono::microseconds(2));
      conditional_variable.wait_until(ulocker, time_point);

      /* Future and Promise */
      std::promise<int> my_promise;
      std::future<int> my_future = my_promise.get_future();

      /* async() */
      std::future<int> fu = std::async(scl::factorial, 6);

      /* packaged task */
      std::packaged_task< int(int) > t(scl::factorial);
      std::future<int> fu2 = t.get_future();
      t(6);

   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}