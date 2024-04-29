#include <cstdlib>
#include <regex>      // use regex_math
#include <iostream>   // std::cin and std::cout
#include <string>     // string.
#include <chrono>     // time and date

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_15.cpp
 */

// Scientific Computing Library
namespace scl
{
}

/*
 * Introduction to chrono C++ 11
 * -- A precision-neutral library for time and date.
 *
 * Clock: 
 *    std::chrono::system_clock: current time according to the system - is not steady.
 *    std::chrono::steady_clock: Goes at a uniform rate.
 *    std::chrono::high_resolution_clock: provides smallest possible tick period.
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   
   {
   std::ratio<2, 10> my_ratio;  // 1/10;
   std::cout << my_ratio.num << "/" << my_ratio.den << std::endl;

   // The priod of the clock is 100 nano-second
   std::cout << std::chrono::system_clock::period::num 
             << "/" 
             << std::chrono::system_clock::period::den 
             << std::endl;
   }

   {


   //std::chroono::duration<>: represents time duration.
   // 2 hours:  a number and a unit.
   // duration<int, ratio<1,1>  // number of seconds stored in a int
   // duration<double, ratio<60,1>   // number of minutes stored in double.
   // nanoseconds, microseconds, milliseconds, seconds, minutes, hours.
   // system_clock::duration -- duration< T system_clock::period>
   
   std::chrono::milliseconds mi(2700);   // 2700 microseconds.
   std::cout << mi.count() << std::endl;
   std::chrono::nanoseconds na = mi;     // 2700 nanoseconds.
   std::cout << na.count() << std::endl; // 2700 000 nanoseconds.

   // 2 miliseconds and 700 microseconds are truncated.
   std::chrono::milliseconds mill = std::chrono::duration_cast<std::chrono::milliseconds>(mi);

   mi = mill + mi; //2000 + 2700 = 4700
   }

   {
   // std::chrono::time_point<>: represents a point of time.
   // 00:00 January 1, 1970 (Corordinated Universal Time - UTC)  -- epoch of a clock.
   // time_point<system_clock>: according to system_clock, the elapsed time since epoch in milliseconds.
   // system_clock::time_point -- time_point<system_clock, system_clock::duration>
   // steady_clock::time_point -- time_point<steady_clock, steady_clock::duration>

      // current time of the system clock.  number of clock cycles from epock till now.
      std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
      std::cout << tp.time_since_epoch().count() << std::endl;
      tp = tp + std::chrono::seconds(2);
      std::cout << tp.time_since_epoch().count() << std::endl;

      std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      std::cout << "I am boared" << std::endl;
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::chrono::steady_clock::duration d = end - start;
      if (d == std::chrono::steady_clock::duration::zero() )
         std::cout << "no time elapsed." << std::endl;

      std::cout << d.count() << std::endl;
      std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(d).count() << std::endl;
   }  

   return 0;
}