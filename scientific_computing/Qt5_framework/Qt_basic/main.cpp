#include <QtCore/QCoreApplication>   // Gives access to QCoreApplication.
#include <QDebug>                    // Gives access to qDebug, qInfo.
#include <QProcessEnvironment>       // Gives access to QProcessEnvironment

#include <array>                     // array from 'STL'
#include <iostream>                  // Gives access to std::cout.
#include <QTextStream>               // Gives access to boombox
#include <iostream>                  // Gives access to 'STL' ???

#include "include/animal.h"                  // gives access to CAnimal
#include "include/feline.h"
#include "include/canine.h"
#include "include/mammel.h"
#include "include/laptop.h"                  // gives access to CLaptop
#include "include/applicance.h"
#include "include/lion.h"
#include "include/destination.h"
#include "include/source.h"
#include "include/radio.h"
#include "include/station.h"
#include "include/test_watcher.h"
#include "include/watcher.h"

void test(int age=0)
{
   qInfo() << "Age: "
           << age;
}

void test(bool is_active)
{
   qInfo() << "is_active: "
           << is_active;
}

// function name is test; returns void; takes the address 0xABCD or 
// referance of a laptop class type; parameter is machine, &machine 
// is 0x1234;  Passing an object by referance (l-value) not by value
// or the (r-value).  Note:  passing an object by value you are 
// are making a copy of that object.  By design, QObject instances
// are not copy-able.
void test(scl::CLaptop & machine)
{
    qInfo() << "test machine: " 
            << machine.name_ 
            << " " 
            << machine.weight_;
    machine.test();
}

// return void, make_latop is 0x1234, parent is a pointer variable
// which is pointing to a QObject class type.
// parent = 0x1234; &parent = 0xABCD; *parent is content 
void make_laptop(QObject * parent)
{
    scl::CLaptop my_laptop(parent, 
                            "laptop_mine");

    scl::CLaptop your_laptop(nullptr, 
                                "laptop_yours");

    my_laptop.weight_ = 3;
    your_laptop.weight_ = 5;

}

void test_scope(int number)
{
    qInfo() << "Number: "
            << number
            << "Address "
            << &number;
}

void test_scopeRef(int &number)
{
    qInfo() << "Number: "
            << number
            << "Address "
            << &number;
    number = 50;
}

// test_address return void, parameters is QString 
// passed by r-value. name = "hello", &name = 0xABCD
void test_address(QString name)
{
    qInfo() << "name is at (stack): " << &name;
}

// test_pointer returns void, parameter is pointer variable
// pointing to an object of class type Qstring
// name = 0x1234, *name = "hello", &name = 0xABCD
void display_pointer(QString * name)
{
    qInfo() << "name is " << *name;
    qInfo() << "name is at " << name;
    qInfo() << "name is at (stack)" << &name;  // The value on stack memory is a copy of data passed on to the function.
}

/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */
int main (int argc, char * argv[])
{
    /*
     Print information onto the screen.
     */
    qInfo() << "INFO: Hello World ...";
    qInfo("INFO: Second Hello World ...");

    QCoreApplication a(argc, argv);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    // QString path = QProcessEnvironment::systemEnvironment().value("pwd");
    QString path = env.value("Path");
    qDebug() << "Path Values: " 
             << path;

    {
        bool isOn = false;
        qInfo() << "is it on: "
                << isOn;

        int age = 44;
        double height = 6.02;
        signed int dogs = -1;
        unsigned int cats = -1;

        const int id = 215;

        age = 44.90;          // implicit conversion from 'double' to 'int'.
        qInfo() << "Age: "
                << age
                << "Height: "
                << height
                << " and bytes size in memory"
                << sizeof(height)
                << "Dogs: "
                << dogs
                << " and byte size in memory: "
                << sizeof(dogs)
                << "Cats: "
                << cats;
    }
    
    {
        // Zero based enum.
        enum Colors {red, green, blue};

        // https://cplusplus.com/doc/tutorial/structures/
        struct product_t
        {
           int weight;
           double value;
           Colors color;
        };

        Colors my_color = Colors::blue;
        qInfo() << "Color: "
                << my_color;

        product_t laptop;
        laptop.value = 1;
        laptop.weight = 2 ;
        laptop.color = Colors::red;

        qInfo() << " sizeof(laptop): "
                << sizeof(laptop);
        qInfo() << "Good Bye ... \n";
    }

    {
        // Array
        int ages[4] = {23, 7, 75, 1000};
        qInfo() << "Ages: "
                << ages
                << " "
                << ages[0]
                << " "
                << ages[1]
                << " "
                << ages[2]
                << " "
                << ages[3];

        // 'STL' template container 'array' holding integers
        std::array<int, 4> cars;
        cars[0] = 23;
        cars[1] = 7;
        cars[2] = 75;
        cars[3] = 1000;  // end of array.
        // cars[4] = 999;
        qInfo() << "cars: "
                << &cars[0]
                << " "
                << cars[0]
                << " "
                << cars[1]
                << " "
                << cars[2]
                << " "
                << cars[3];

        qInfo() << "Size: " 
                << cars.size()
                << "Size of: "
                << sizeof(cars)
                << "Last: " 
                << cars[ cars.size() - 1 ];
    }

    {
        int age = 44;
        std::cout << "Hello World from STL ... " << std::endl;
        std::cout << "Age: " << age << std::endl;

        qInfo() << "Hello World from qInfo ....";

        std::cerr << "Error: out " << std::endl;
        qDebug() << "Qt Debug ...";
        qCritical() << "Qt Critical ...";
        // qFatal("Qt Fatal ...");
        qInfo() << "Good Bye ... \n";
    }

    {
        // overload a function in C++17
        test();
        test(false);
    }

    {
        // Pass by Referance and by Value
        class CTest
        {
        public:
           int my_int_;
        public:
           // pass by value or COPY, so when the program counter (PC) leaves the 
           // scope of this function that COPY of int data type that is created
           // on the stack is freed.
           void test_value(int some_int)
           {
                some_int = some_int * 10;
                qInfo() << "some_int: " << some_int;
           }
           // return void, test value, pass an integer by address or referance 
           // the address of some_int == &some_int == 0x1234 is passed to the
           // function member.  The actual object is pass not a COPY.  Passing by
           // Ref is not a pointer.  
           void test_referance(int& some_int)
           {
                some_int = some_int * 10;
                qInfo() << "some_int: " << some_int;
           }
        };

        CTest temp_test;
        int x = 5;
        qInfo() << "Testing pass by value";
        temp_test.test_value(x);
        temp_test.test_value(5);

        qInfo() << "Testing pass by referance";
        temp_test.test_referance(x);

        // Scope means brackets {}, the variables, objects, l-values 
        // are not defined outside of the scope or brackets {}.
        // qInfo left shift operator
        qInfo() << "Good Bye ... \n";
    }

    {
        struct Laptop_t
        {
           int weight;
           double as_kilogram()
           {
               return weight * 0.453592;
           };
        };
        
        Laptop_t notebook;
        double my_double;
        notebook.weight = 5.0;
        my_double = notebook.as_kilogram();
        qInfo() << "My Double: " << my_double;
        qInfo() << "Good Bye ... \n";
    }
 
    {
        // classes. 
        scl::CAnimal cat;
        scl::CAnimal dog;
        scl::CAnimal bird;

        cat.speak("meow");
        cat.speak("bark");
        cat.speak("caw");
        qInfo() << "Good Bye ... \n";
    }

   {
        // classes. 
        // a is an object of class type QCoreApplication, &a == 0x1234
        scl::CLaptop my_laptop(&a, 
                               "laptop_mine");

        scl::CLaptop your_laptop(nullptr, 
                                 "laptop_yours");

        my_laptop.weight_ = 3;
        your_laptop.weight_ = 5;

        make_laptop(&a);
        test(my_laptop);

        qInfo() << "Good Bye ... \n";
    }

    {
        qInfo() << "\n";
        scl::CFeline kitty;
        scl::CCanine puppy;
        scl::CMammel mammel;

        qInfo() << "Good Bye ... \n";
    }

    {
        scl::CApplicance kitchen_5000;
        qInfo() << "Can cook" << kitchen_5000.cook();
        qInfo() << "Can Grill" << kitchen_5000.grills();
        qInfo() << "Can Freeze" << kitchen_5000.freeze();

        qInfo() << "Good Bye ... \n";
    }

    {
        qInfo() << "\n";
        qInfo() << "\n";
        scl::CLion simba;
        simba.speak();
        qInfo() << "Good Bye ... \n";
    }

    {
        qInfo() << "\n";
        qInfo() << "\n";
        int number = 75;
        qInfo() << "number: " << number << "address " << &number;
        test_scope(number);
        test_scopeRef(number);
        qInfo() << "number: " << number << "address " << &number;
 
        qInfo() << "Good Bye ... \n";
    }

    {
        qInfo() << "\n";
        qInfo() << "\n";
        QString name = "Bryan";

        qInfo() << "Address of a: " << &a;
        scl::CAnimal2 cat(&a, "cat_fluffy");
        scl::CAnimal2 dog(&a, "dog_fido");

        qInfo() << "My name is at " << &name;
        scl::CAnimal2 person(&a, name);
        person.say_hello("What is up");

        qInfo() << "Good Bye ... \n";
    }

    {
        qInfo() << "\n";
        qInfo() << "\n";
        QString name = "Bryan";

        qInfo() << "Name is at " << &name;

        qInfo() << "Good Bye ... \n";
    }

    {
        qInfo() << "\n";
        qInfo() << "\n";
        QString name = "Bryan";

        qInfo() << "length: " << name.length();

        qInfo() << "Name: " << name << " is at: " << &name;
        test_address(name);
        display_pointer(&name);

        // pointer object pointing to a QString class type.
        // ptr_description == 0x1234; *ptr_description == "hello"; &ptr_description == 0xABCD
        QString * ptr_description = new QString("hello");
        qInfo() << "length: " << ptr_description->length();
        qInfo() << "content ptr_description: " << *ptr_description;
        qInfo() << "pointer value ptr_description: " << ptr_description;
        qInfo() << "address ptr_description: " << &ptr_description;

        qInfo() << "Actual length: " << ptr_description->length();
        qInfo() << "Actual size: " << sizeof(ptr_description);
        qInfo() << "ptr_description: " << (*ptr_description);
        qInfo() << "sizeof ptr_description: " << sizeof(*ptr_description);


        display_pointer(ptr_description);
        delete ptr_description;
        ptr_description = nullptr;

        // Every time a r-value is passed on by value, C++ copies the data on to the stack memory.
        // Every time a r-value is passed on by referance, C++ passed the address to the function.
        qInfo() << "Good Bye ... \n";
    }

    {
        qInfo() << "\n";
        qInfo() << "\n";

        QString name = "Bryan";

        qInfo() << "Address of a: " << &a;
        scl::CAnimal2 cat(&a, "cat_fluffy");
        scl::CAnimal2 dog(&a, "dog_fido");

        qInfo() << "My name is at " << &name;
        scl::CAnimal2 person(&a, name);
        person.say_hello("What is up");

        scl::CAnimal2 lion(0, "lion");
        scl::CAnimal2 tiger(nullptr, "lion");
        lion.say_hello("what is up");
        tiger.say_hello("what is up");

        scl::CAnimal2 * bobcat = new scl::CAnimal2(0, "bobcat");
        bobcat->say_hello("what is up");
        
        delete bobcat;
        bobcat = nullptr;

        qInfo() << "Good Bye ... \n";
    }

    {
        qInfo() << "\n";
        qInfo() << "\n";
 
        scl::CSource source_object;
        scl::CDestination destination_object;
        
        QObject::connect(&source_object, 
                         &scl::CSource::my_signal, 
                         &destination_object, 
                         &scl::CDestination::on_message);

        source_object.test();
        qInfo() << "Good Bye ... \n";        
    }

    {
        qInfo() << "\n";
        qInfo() << "\n";
        scl::CRadio boombox;
        scl::CStation * channels[3];
        channels[0] = new scl::CStation(&boombox, 94, "Rock and Roll");
        channels[1] = new scl::CStation(&boombox, 101, "Hip Hop");
        channels[2] = new scl::CStation(&boombox, 85, "News");

        // Qt::QueuedConnection = used when you work wiht threads.
        QObject::connect(&boombox,
                         &scl::CRadio::quit,
                         &a,
                         &QCoreApplication::quit,
                         Qt::QueuedConnection);
        do 
        {
            /* Remove Break to use*/
            break;

            qInfo() << "Enter on, off, test or quit";
            QTextStream qtin( stdin );
            QString line = qtin.readLine().trimmed().toUpper();

            if (line == "ON")
            {
                qInfo() << "Turning Radio ON";
                for(int i = 0; i < 3; i++)
                {
                    scl::CStation * channel = channels[i];
                    QObject::connect(channel, 
                                     &scl::CStation::send, 
                                     &boombox, 
                                     &scl::CRadio::listen);
                }
                qInfo() << "Radio is on";
            }

            if (line == "OFF")
            {
                qInfo() << "Turning Radio OFF";
                for(int i = 0; i < 3; i++)
                {
                    scl::CStation * channel = channels[i];
                    QObject::disconnect(channel, 
                                     &scl::CStation::send, 
                                     &boombox, 
                                     &scl::CRadio::listen);
                }
                qInfo() << "Radio is OFF";

            }

            if (line == "TEST")
            {
               qInfo() << "Turning Radio TEST";
                for(int i = 0; i < 3; i++)
                {
                    scl::CStation * channel = channels[i];
                    channel->broadcast("BroadCasting live: ");
                }
                qInfo() << "Radio is Testing";
            }

            if (line == "QUIT")
            {
               qInfo() << "Turning Radio QUIT";
               emit boombox.quit();
               qInfo() << "Radio is Quiting";
               break;
            }

        } while (true);

        qInfo() << "Good Bye ... \n";        
    }

    {
        qInfo() << "\n";
        qInfo() << "\n";

        scl::CTestwatcher source_testWatcher;
        scl::CWatcher destination_watcher;

        QObject::connect(&source_testWatcher,
                         &scl::CTestwatcher::message_changed,
                         &destination_watcher,
                         &scl::CWatcher::message_changed);

        source_testWatcher.setProperty("message", "hello world!");
        source_testWatcher.set_message("testing");
        qInfo() << "Good Bye ... \n";        
    }

    {
        /*
           Dynamic Cast (Cast a spell)
           Dynamic cast can be used only with pointers and referances to objects.
           It is intended to make sure the result of the type conversion (casting)
           is valid and complete.
         */
        qInfo() << "\n";
        qInfo() << "\n";

        double value = 43.88;
        qInfo() << "Double " << value;
        // Implicit conversion - should not be trusted.
        int age = value;
        qInfo() << "int - implicit " << age;
        // Explicit conversion shoud be trusted.  Casting old style.
        age = (int)value;
        qInfo() << "int - explicit" << age;
 
        qInfo() << "Good Bye ... \n";        
    }

    return a.exec();
}