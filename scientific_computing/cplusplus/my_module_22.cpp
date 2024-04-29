#include <cstdlib>
#include <regex>      // use regex_math
#include <iostream>   // std::cin and std::cout
#include <string>     // string.
#include <chrono>     // time and date
#include <random>     // Random Number Generation.
#include <tuple>      // tuple

/*
 * g++ -std=c++17 -g -o ./run_compute my_module_19.cpp
 */

// Scientific Computing Library
namespace scl
{
class B
{
public:
   void f_pub() {std::cout << "f_pub is called...  \n"; }
protected:
   void f_prot() {std::cout << "f_prot is called... \n";}   
private:
   void f_priv() {std::cout << "f_priv is called... \n";}
};  //end of class B

class D_pub : public B
{
public:
   void f()
   {
      B::f_pub();   // O.K. D_pub's public function.
      B::f_prot();  // O.K. D_pub's protected function.
      //B::f_priv();  // Error. B's private function.
   }
};  // end of class D_pub

class D_prot : protected B
{
public:
   using B::f_pub;
   void f()
   {
      B::f_pub();   // O.K. D_pub's protected function.
      B::f_prot();  // O.K. D_pub's protected function.
      //B::f_priv();  // Error. B's private function.
   }
};  // end of E_prot

class D_priv : protected B
{
public:
   void f()
   {
      B::f_pub();   // O.K. D_pub's private function.
      B::f_prot();  // O.K. D_pub's private function.
      //B::f_priv();  // Error. B's private function.
   }
};   // end of D_priv

/*
 * public inheritance: is-a relation, eg. D_pub is-a kind of a B.
 * private inheritance: similar to has-a relation.
 */
class ring
{
   virtual void tremble(){};
   public:
   void tinkle() 
   {
      std::cout << "ring: " << std::endl;
      this->tremble();
   }
}; //end of ring

/* composition */
class dog
{
   private:
   ring my_ring;
   public:
   void tinkle() 
   { 
      std::cout << "dog: " << std::endl;
      // call forwarding ...
      my_ring.tinkle();
   }
   // virtual void bark() 
   // { 
   //    std::cout << "I am just a dog: " << std::endl;
   // }
   virtual void bark(std::string msg = "just a")
   {
      std::cout << "Whoof, ia am " << msg << " dog." << std::endl;
   }
   void bark(int age) 
   {
      std::cout << "I am " << age << " years old" << std::endl;
   }

}; //end of dog.

/* private inheritance */
class private_dog : private ring
{
   virtual void tremble();
   public:
   using ring::tinkle;

}; //end of private_dog.

class yellow_dog : public dog
{
   public:
   using dog::bark;
   // virtual void bark()
   // {
   //    std::cout << "I am just a yellow_dog ..." << std::endl;
   // }
   virtual void bark(std::string msg = "yellow")
   {
      std::cout << "Whoof, ia am " << msg << " dog." << std::endl;
   }

};  // end: yellow_dog


/*
 * public inheritance => 'is-a' relation.
 * A derived class should be able to do everything the base class can do.
 */
class bird
{
public:
      virtual void fly()=0;
};  // end of bird

class flyable_bird : public bird
{
public:
      virtual void fly()
      {
         std::cout << "flyable_bird: " << std::endl;
      };
};  // end of flyable_bird

class penguin : public bird
{

};  // end of penguin.

class Engineer
{
   public:
   // 40 APIs
};  // End: Engineer

class Son
{
   public:
   // 50 APIs
};  // End Son

class Andy : public Engineer, Son
{
   public: 
   // 550 APIs
};  // End: Andy


}  // end: namespace scl
/*
 * They specifies difference access control from the drived class to the base class.
 * Access Control:
 * 1. None of the derived classes can access anythng that is private in the base B.
 * 2. D_pub inherits public members of B as public and protected members of B as protected.
 * 3. D_priv inhertis the public and protected members of B as private.
 * 4. D_prot inhertis the public and protected members of B as protected.

 * Casting:
 * 1. Anyone can cast a D_pub* to B*. D)_pub is a special kind of B.
 * 2. D_priv's members and friends can cast a D_priv* to B*.
 * 3. D_prot's members, friends and and children can cast a D_prot* to B*.

 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;

   {
      scl::D_pub d1;
      d1.f_pub();           // O.K. f_pub() is D_pub's public function member.

      scl::D_prot d2;

      // error. f_pub() is D_prot's protected method.  
      // must add  using 'B::f_pub' in the class definition.
      d2.f_pub();            

      scl::B * ptr_B = &d1;     // O.K.
      // ptr_B = &d2;            // error.
   }

   {
      // Namespace scl has a class 'yello_dog' the ptr_yellowDog is a pointer object with location 0x1234
      // *ptr_yellowDog is temporaty data r-value.  &ptr_yellowDog 0xABCD.
      scl::yellow_dog * ptr_yellowDog = new scl::yellow_dog();

      // the pointer to the object ptr_yellowDog has a method defined and using the arrow operator.      
      // bark should be a virtual method of class dog.
      ptr_yellowDog->bark();
      ptr_yellowDog->bark(5);

      // this is inhertance and class dog should be the parent of yellowdog.  
      // ptr_dog is a poninter object of type dog with l-value 0x1234 *ptr_dog is the r-value
      // temporary data, &ptr_dog
      scl::dog * ptr_dog = ptr_yellowDog;

      // if bark is not defined as virutal the parent's bark is invoked.
      // do not over ride none-virutal functions.  if there is a function
      // that you want to override, make that function virtual.
      ptr_dog->bark();
      ptr_dog->bark(8);
      std::cout << "Good Bye" << std::endl;
   }
   /*
    * Summary:
    * 1. precise definition of classes.
    * 2. do not override non-virtual functions.
    * 3. do not override default parameters values for vitrual functions.
    * 4. force inheritance of shawdow functions.
    */

   {
      /*
       * Interface Segregation Principal.
       * Split large interfaces into smaller and more specific ones to that clients
       * only need to know about the methods that are of interest to them.
       */

      /*
       * Pure Abstract Classes.
       * Abstract Class:  A class has one or more pure virtual functions.
       * 
       * Pure Abstract classes:
       * A class containing only pure virtual functions.
       *   - no data
       *   - no concrete functions.
       */
      
      /*
       * Summary:
       * 1. Multiple inheritance is an important techniqe, e.g. ISP
       * 2. Derived only from PACs when using mutlitple inheritance.
       */

      /*
       * The duality of public inheritance.
       * 1. Inheritance of interface.
       * 2. Inheritance of implementation.
       */

      class Dog
      {
       public:
           // Pure virtual function.
           virtual void bark() = 0;
           // Regular non-virtual function member.
           // The derived class yellow_dog should NOT over-ride this method.
           // The yellow_dog rather inherits the implementation of this function.
           void run()
           {
               std::cout << "I am just a Dog ..."  << std::endl;
           }
           // Virtual function member.
           // The derived yellow_dog class can inhert the implementation and the interface.
           virtual void eat()
           {
               std::cout << "Dog is eating ..."  << std::endl;
           }
       protected:
           void sleep() 
           {
               std::cout << "Dog is sleeping ..."  << std::endl;
           }

      };  // End: Dog

      class yellow_dog : public Dog
      {
       public: 
           // Derived class over-rides the funtion member of the base class
           // yellow_dog inherit the interface of the dog's function member.
           // It does not inherit the implementation of the bug function.
           virtual void bark()
           {
              std::cout << "I am a yellow dog" << std::endl;
           }
           // Since the derived class yellow_dog implemented it's own version,
           // it only inherits the interface.
           virtual void eat()
           {
              std::cout << "yellow_dog is eating ..." << std::endl;
           }
           void iSleep()
           {
              std::cout << "yellow_dog is sleeping ..." << std::endl;
              Dog::sleep();
           }

      };  // End: yellow_dog

      Dog * my_dog = new yellow_dog();
      my_dog->bark();
      my_dog->run();
      my_dog->eat();
      // my_dog->sleep();
      delete my_dog;
      my_dog = nullptr;

      yellow_dog * my_yellowDog = new yellow_dog();
      my_yellowDog->iSleep();
      delete my_yellowDog;
      my_yellowDog = nullptr;
      std::cout << "Good Bye" << std::endl;
   }

   {
      /*
       * Interface Inheritance.
       * 1. sub-typing.
       * 2. ploymorphism
       * Example: virtual void some_function() = 0;
       */
      
      /*
       * Implementation Inheritance.
       * 1. increase code complexity.
       * 2. Not encouraged.
       * Example: 
       * public:
       *    void run() { std::cout << "I am running ..."; }
       *    virtual void eat() { std::cout << "I am eating ..."; }
       * protected:
       *    void sleep() { std::cout << "I am sleeping ..."; }
       */

      /*
       * Guidelines for Implementation Inheritance:
       * 1. Do not use inheritance for code reuse, use composition.
       * 2. Minimize the implementation in base class, base class should be thin.
       * 3. Minimize the level of hierarchies in implementation inheritance.
       */


      std::cout << "Good Bye" << std::endl;
   }

   {
      /*
       * Code reuse: Inheritance vs Composition.
       */

      class CDog
      {
         // Code reuse with inheritance.  All extended dogs have the 
         // following data members and function members.
         std::string skin_color;
         virtual void run() = 0;
         virtual void bark() = 0;
         virtual void eat() = 0;
         virtual void sleep() = 0;
          // Common Activities.
      };  // End: CBaseDog

      class CBullDog : public CDog
      {
          // Call the common activities to perform more tasks.
      };  // End: CBullDog

      class CShepheredDog : public CDog
      {
          // Call the common activities to perform more tasks.
      };  // End: CShepheredDog

      std::cout << "Good Bye" << std::endl;
   }

   {
      /*
       * Code reuse: Composition is better than inheritance
       * 1. Less code coupling between reused code and reuser of the code.
       *    a. Child class automatically inherits ALL parent class's public members.
       *    b. Child class can access parent's protected members.
       *       - Inheritance breaks encapsulation. 
       * 2. Dynamic Binding
       *    a. Inheritance is bound at compile time.
       *    b. Composition can be bound either at compile time or at run time.
       */

      class CActivityManager
      {

      };  // End: CActivityManager
        
      class CDog
      {
      };  // End: CBaseDog

      class CBullDog : public CDog
      {
         CActivityManager * prt_activityManager;
          // Call the common activities throught ptr_activityManager
      };  // End: CBullDog

      class CShepheredDog : public CDog
      {
         CActivityManager * prt_activityManager;         
          // Call the common activities throught ptr_activityManager
      };  // End: CShepheredDog

      std::cout << "Good Bye" << std::endl;
   }

   std::cout << "Good Bye" << std::endl;
   return 0;
}