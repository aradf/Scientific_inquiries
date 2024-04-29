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

// stl template deque contains stl's packaged_task template.
// the package_task takes on function object returning int.

// typedef std::packaged_task< int() > my_taskedDeque;
// std::deque< my_taskedDeque > tasked_deque;

}  // end: namespace scl

// void thread_1()
// {
//    std::packaged_task< int() > t;
//    t = std::move(scl::tasked_deque.front());

//    t();
// }

/*
 * Packaged Task.
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   std::cout << "main thread id: " << std::this_thread::get_id() << std::endl;
   std::cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;

   {
      // std::thread t1(thread_1);
      // t is a function object.
      auto t = std::bind(scl::factorial, 6);

      // scl::tasked_deque.push_back(t);

      // t1.join();
      
      t();
      std::cout << "Good Bye" << std::endl;   
   }

   {
      // stl template packaged_task contains int.  Wrapps any callable function, lambda expression.
      // function object.  'int(int)' is a function that takes integer and returns integer like
      // factorial.
      std::packaged_task< int(int) > t1(scl::factorial);

      // stl packaged_task template containing a function object. 'int(int)' means the function
      // object takes an integer and returns an integer.  Note the stl bind function takes on the
      // factoral function object and the integer seven passed on to it.
      std::packaged_task< int() > t2(std::bind(scl::factorial, 7));


      std::thread t3(scl::factorial, 6);

      /// ... do something.

      // Because of stl packaged_task template containing a function object, t1 is 
      // invoked in a different context where it is created.
      t1(6);
      int x = t1.get_future().get();

      t2();

      std::cout << "Good Bye" << std::endl; 
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}