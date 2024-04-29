#include <cstdlib>
#include <regex>      // use regex_math
#include <iostream>   // std::cin and std::cout
#include <string>     // string.

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_19.cpp
 */

// Scientific Computing Library
namespace scl
{

/*
 * Member Operator new
 */
class dog
{
   std::string dog_name_;
   int hair[10000000000L];
   std::new_handler origHandler;
   public:
   dog() {}
   ~dog() {}
   static void NoMemForDog()
   {
      std::cerr << "No more memmory for doggy, bo. " << std::endl;
      throw std::bad_alloc;
   }
   
   void* operator new(const std::size_t size) 
   {
      origHandler = std::set_new_handler(NoMemForDog);
      void * pv = ::operator new(size);
      std::set_new_handler(origHandler);
      return pv;
   }

   static void operator delete(void * ptr_memory) 
   {
      std::cout << "Bo is deleting a dog ... \n" << std::endl;
      custome_newForDog();
      free(ptr_memory);
   }

   void custome_newForDog(std::size_t size)
   {
      std::cout << "custome_newForDog ... \n" << std::endl;
   }

};  // end: dog

class yellow_dog : public dog
{
   int age;
   public:
   static void* operator new(const std::size_t size)
   {

   }
   static void operator delete(void * ptr_memory) 
   {
      std::cout << "Bo is deleting a yellow dog ... \n" << std::endl;
      custome_newForDog();
      free(ptr_memory);
   }


};  //end; yellow_dog

}  // end: namespace scl

/*
 * Dmystifying operator new and delete
 * What happens when the following code is executed ...
 * 
 * dog * ptr_dog = new dog();
 * 1. Operator new is invoked to allocate memory.
 * 2. dog's constructor is invoked to create an instance of dog.
 * 3. if step(2) throws an exception, call operator delete to free
 *    the allocated memory in step (1)
 * 
 * delete ptr_dog
 * 1. dog's destructor is called.
 * 2. operator delete is called to free memory.
 */

/*
 * Simplified version of operator new
 * return a pointer to void.  
 * The operator keyword tells the * the compiler this code over-rides new.
 * object type std::size_t is a paramete.
 * throw an exception std::bad_alloc.
 * Note: New handler is a function invoked when operator new failed to allcoate memory.
 *       set_new_handler() installs a new handler and returns current new handler.
 */
// void* operator new(std::size_t size) throw(std::bad_alloc)
void* operator new(std::size_t size)
{
      while(true)
      {
          //allocate memory.
          void* ptr_memory = malloc(size);
          if (ptr_memory != nullptr)
          {
            // return memory if succesful.
            return ptr_memory;
          }

          // get new handler.
          std::new_handler Handler = std::set_new_handler(0);
          std::set_new_handler(Handler);

          if (Handler != nullptr)
          {
              // invoke new handler.
              (*Handler)();
          }
          else
              {
                 // if the handler is null pointer throw std::bad_alloc exception.
                 throw std::bad_alloc();
              }
      }
}

/*
 * Why do we want to customize new/delete.
 * 1. Usage error detection:
 *    - Memory leak detection/garbage collection.
 *    - Array index overrun/underrun.
 * 2. Improve efficiency:
 *    a. Clustering related objects to reduce page fault.
 *    b. Fixed size allocation (good for appication with many small objects).
 *    c. Align simialr size objects to same places to reduce fragmentation.
 * 3. Perform additional tasks.
 *    a. Fill the deallocated memory with 0's - security.
 *    b. Collect usage statistics.
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   scl::yellow_dog * ptr_yellowDog = new scl::yellow_dog();
   delete ptr_yellowDog;

   std::cout << "Good Bye" << std::endl;
   return 0;
}