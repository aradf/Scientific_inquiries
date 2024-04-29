/*
 * STL Headers
 */
#include <cstdlib>
#include <iostream>   // std::cin and std::cout
#include <fstream>    // std::ofstream
#include <thread>     // std::thread.

/*
 * g++ -std=c++17 -pthread -g -o ./run_compute my_module_41.cpp
 */

// Scientific Computing Library
namespace scl
{
   class A
   {
   public:
      void f(int x, char c)   {}
      long g(double x) { return 0; }
      // return an integer, '()' operator is over-loaded
      // A is a functor.
      int operator()(int n) {return 0;}
   };

   void foo(int x)
   {

   }


}  // end: namespace scl

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   std::cout << "main thread id: " << std::this_thread::get_id() << std::endl;
   std::cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;

   {
      // create an instance of A.
      scl::A a;

      // stl class type thread initialized with callable object a.
      // copy of object a and invoke it in a working thread.
      std::thread t1(a, 6);

      // stl class type thread initialized with a stl ref wrapper 
      // callable object is used in working thread: note does not 
      // make a copy of a.
      std::thread t2(std::ref(a), 6);

      // do not make a copy of ojbect a and just move the object a
      // from main thread to the working thread.
      // a is no longer useable in the main thread.
      std::thread t8(std::move(a), 6);

      // temp A is created and invoked the working thread.
      std::thread t3(scl::A(), 6);

      std::thread t4([](int x){ return x*x; }, 6);

      std::thread t5(scl::foo, 7);

      // stl class type thread initialized, referance or address of class A's 
      // function member f, callable object a (make copy of a.f(8, 'w')) in a 
      // different thread, 
      std::thread t6(&scl::A::f, a, 8, 'w');

      // do not make a copy of object a and just pass it by referance or address.
      std::thread t7(&scl::A::f, &a, 8, 'w');

      /*
      // stl function async: function signature stl::launch::async callable object a.
      std::async(std::launch::async, a, 6);

      // STL function bind: function signature callable object a,
      std::bind(a, 6);
      std::call_once(once_flag, a, 6);
      */
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}