#include <cstdlib>
#include <regex>      // use regex_math
#include <iostream>   // std::cin and std::cout
#include <string>     // string.
#include <chrono>     // time and date
#include <random>     // Random Number Generation.
#include <tuple>      // tuple

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_16.cpp
 */

/*
 * Random Engine:
 * Stateful generator that generates random values within predefined min and max.
 * Not truely rnadom - pseudo-random 
 */

// Scientific Computing Library
namespace scl
{
   void print_random(std::default_random_engine some_randomEngine)
   {
      for (int iCnt = 0; iCnt < 10; iCnt++)
      {
         std::cout << some_randomEngine() << " ";
      }
      std::cout << std::endl;
   }

   std::tuple<std::string, int> return_nameAge()
   {
      std::tuple<std::string, int> t1;
      t1 = std::make_tuple("Bob", 50);
      return (t1);
   }
}

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   {
   std::default_random_engine random_enginGenerator;
   std::cout << "Min: " 
             << random_enginGenerator.min() 
             << std::endl;

   std::cout << "Max: " 
             << random_enginGenerator.max() 
             << std::endl;
   
   std::cout << random_enginGenerator() 
             << std::endl;

   std::cout << random_enginGenerator() 
             << std::endl;

   std::stringstream current_state;
   current_state << random_enginGenerator;   // save current state 

   std::cout << random_enginGenerator() 
             << std::endl;

   std::cout << random_enginGenerator() 
             << std::endl;

   current_state >> random_enginGenerator;   // restore the saved state 

   std::cout << random_enginGenerator() 
             << std::endl;

   std::cout << random_enginGenerator() 
             << std::endl;

   }


   {
   std::default_random_engine e1;
   std::default_random_engine e2;
   scl::print_random(e1);
   scl::print_random(e2);
   
   unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
   std::default_random_engine e3(seed);
   scl::print_random(e3);
   
   e1.seed();    // reset engine e1 to initial state.
   e1.seed(109); // set engine to a seed according to seed 109
   e2.seed(109); // set engine to a seed according to seed 109
   if (e1 == e2)
   {
      std::cout << "e1 and e2 have the same state " << std::endl;
   }

   std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7 ,8, 9};
   std::shuffle(vector.begin(), vector.end(), std::default_random_engine());
   // www.cplusplus.com has a table of different random number generators.
   }
   
   {
      unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
      std::default_random_engine e(seed);

      std::cout << e() << std::endl;  // rnage of e.min() to e.max()

      // range: [0, 5];
      /*
       * 1. Bad quality of randomness.
       * 2. Can only provide uniform distribution.
       */
      std::cout << e() % 6 << std::endl;
      
      // generate random number in [0, 5]
      std::uniform_int_distribution<int> uniform_integerDistribute(0,5);
      std::cout << uniform_integerDistribute(e) << std::endl;

      std::uniform_real_distribution<double> uniform_realDistribute(0,5);
      std::cout << uniform_realDistribute(e) << std::endl;

      std::poisson_distribution<int> poisson_distribute(1.0); // mean
      std::cout << poisson_distribute(e) << std::endl;
      
      std::cout << "Normal Distribution ..." << std::endl;
      std::normal_distribution<double> normal_distribute(10.0, 3.0);  // mean and standard deviation.
      std::vector<int> vector(20);

      for (int iCnt = 0; iCnt < 800; iCnt++)
      {
         int num = normal_distribute(e);                     // convert double to integer.
         if (num >= 0 && num < 20)
         {
            vector[num]++;   // e.g. v[10] records number of times 10 appeared.
         }
      }
      for (int iCnt = 0; iCnt < 20; iCnt++)
      {
         std::cout << iCnt << ": " << std::string(vector[iCnt], '*') << std::endl;
      }
      std::cout << "Good Bye" << std::endl;   

   }

   {
      std::pair<int, std::string> ptr_pair = std::make_pair(23, "hello");
      std::cout << ptr_pair.first << " " << ptr_pair.second << std::endl;

      std::tuple<int, std::string, char> ptr_tuple(32, "penny wise", 'a');
      std::cout << std::get<0>(ptr_tuple) << " ";
      std::cout << std::get<1>(ptr_tuple) << " ";
      std::cout << std::get<2>(ptr_tuple) << " ";
      std::cout << std::endl;

      // std::get<0>(ptr_tuple) returns a reference to the tuple object.
      std::get<0>(ptr_tuple) = 15;
      std::get<1>(ptr_tuple) = "Pound Foolish";
      std::get<2>(ptr_tuple) = 'A';

      std::cout << std::get<0>(ptr_tuple) << " ";
      std::cout << std::get<1>(ptr_tuple) << " ";
      std::cout << std::get<2>(ptr_tuple) << " ";
      std::cout << std::endl;

      std::string & string_tuple = std::get<1>(ptr_tuple);
      string_tuple = "patience is virtue";

      std::cout << std::get<0>(ptr_tuple) << " ";
      std::cout << std::get<1>(ptr_tuple) << " ";
      std::cout << std::get<2>(ptr_tuple) << " ";
      std::cout << std::endl;
  
      //std::get<3>(ptr_tuple) would not compile, since out of range.
      std::vector<int> vector(3);

      // The subscript operator returns a referance to the elements of the vector as well.
      vector[1] = 4;
      int &i = vector[1];

      std::cout << vector[1] << " " << i << std::endl;
      i = 5;
      std::cout << vector[1] << " " << i << std::endl;

      int iCnt = 1;
      // get<iCnt>(ptr_tuple)  will not compile, since iCnt is not a compile time constant.

      std::tuple<int, std::string, char> t1;   // the t1 tuple object is initialized with default constructor.
      t1 = std::tuple<int, std::string, char>(12, "curiosity kills the cat ", 'd');
      std::cout << std::get<0>(t1) << " ";
      std::cout << std::get<1>(t1) << " ";
      std::cout << std::get<2>(t1) << " ";
      std::cout << std::endl;

      t1 = std::make_tuple(12, "curiosity kills the cat ", 'd');
      std::cout << std::get<0>(t1) << " ";
      std::cout << std::get<1>(t1) << " ";
      std::cout << std::get<2>(t1) << " ";
      std::cout << std::endl;

      // lexicographical comparison.
      if (t1 < ptr_tuple)
         std::cout << "tuple object has comparison overloaded." << std::endl;
      t1 = ptr_tuple;   // member by member copy.

      std::cout << "Good Bye" << std::endl;
   }

   {
      // tuple objects in c++ can store referances.
      std::string string_tuple = "in for a pound";
      std::tuple<std::string &> t3(string_tuple);
      std::get<0>(t3) = "in for a pound";
      t3 = std::make_tuple(std::ref(string_tuple));
      std::cout << "Good Bye" << std::endl;
      std::tuple<int, std::string, char> t2 = std::make_tuple(12, "Curiosity Kills the cat", 'd');
      int x;
      std::string y;
      char z;
      std::make_tuple(std::ref(x), std::ref(y), std::ref(z)) = t2;
      std::tie(x, y, z) = t2;  // doing the same thing.
      std::tie(x, std::ignore, z) = t2;

      auto t4 = std::tuple_cat(t2, t3); //t4 is a tuple<int, string, char>

      // type traits
      std::cout << std::tuple_size<decltype(t4)>::value << std::endl;  // 4
      std::tuple_element<1, decltype(t4)>::type d;                     // d is a string.
   }

   {
      // tuple objects in c++ is like structure.
      struct Persons {std::string name_; int age_;} person;
      person.age_ = 25;
      person.name_ = "Hello";
     
      std::cout << person.name_ <<  " " << person.age_ << std::endl;
      std::tuple<std::string, int> person_tuple;

      std::get<0>(person_tuple) = "world";
      std::get<1>(person_tuple) = 30;

      std::cout << std::get<0>(person_tuple) 
                << " " 
                << std::get<1>(person_tuple) 
                << std::endl;

      // one time only structue to transfer a group of data.
      std::tuple<std::string, int> temp_tuple = scl::return_nameAge(); 
      std::string name;
      int age;
      std::tie(name, age) = scl::return_nameAge();

      // comparison of tuples
      std::tuple<int, int, int> time1, time2;   // hours, minutes, seconds
      if (time1 > time2)
         std::cout << "time1 is a later time2";

      // Multi Index Map.
      std::map<std::tuple<int, char, float>, std::string> multi_indexMap;   
      multi_indexMap[std::make_tuple(2, 'a', 2.3)] = "Fait move mountains.";
      
      // std::unordered_map<std::tuple<int, char, float>, std::string> hash_table;
      int a, b, c;
      std::tie(b,c,a) = std::make_tuple(a, b, c);

      std::cout << "Good Bye" << std::endl;
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}