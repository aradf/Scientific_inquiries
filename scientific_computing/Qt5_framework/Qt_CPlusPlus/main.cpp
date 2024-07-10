#include <QCoreApplication>
#include <QDebug>
#include <QPointer>
#include <QScopedPointer>

#include <memory>                // Gives access to shared pointer.
#include <iostream>              // Gives access to std::err;

#include <../include/first.h>
#include <../include/dog.h>
#include <../include/shared_dog.h>
#include <../include/test.h>


void useit(scl::CTest * ptr_test)
{
    if (!ptr_test)
        return;
    
    qInfo() << "Using: " << ptr_test;
};


/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    {
         auto foo_rawPointer = [] ()
         {
            // ptr_dog is pointer object pointing to CDog class type.
            // ptr_dog == 0x1234 (l-value); *ptr_dog is temp data 
            // content of CDog; &ptr_dog == 0xABCD; '->' operator
            scl::CDog * ptr_dog = new scl::CDog("Tank");
            // ..

            ptr_dog->bark();
            
            if (ptr_dog != nullptr)
            {
                delete ptr_dog;
                // ptr_dog is NOT dangling pointer - defined behavior.
                // candidate for a memory leak;
                ptr_dog = nullptr;
            }
         };

         auto foo_sharedPointer = [] ()
         {
            // STL's templated container holding 'CDog' class type; 
            // ptr_dog.get() == 0x1234;
            std::shared_ptr<scl::CDog> ptr_dog(new scl::CDog("Gunner"));    //count = 1;
            qInfo() << "use_count: " << ptr_dog.use_count();
            ptr_dog->bark();

            {
                std::shared_ptr<scl::CDog> p2 = ptr_dog;            //count = 2;
                qInfo() << "use_count: " << ptr_dog.use_count();
                qInfo() << "use_count: " << p2.use_count();
                p2->bark();
            }
            qInfo() << "use_count: " << ptr_dog.use_count();
            qInfo() << "got it...";

         };  //count = 0;

         auto foo_mixedPointersBADBADBAD = [] ()
         {
            // BADBADBAD
            // An Pointer Object should be assigned to a smart pointer as soon as 
            // it is created.  Raw pointers should not be used ...
            scl::CDog * ptr_dog = new scl::CDog("Tank");

            {
                // STL's templated container 'shared_ptr' holds CDog;
                // shared_dog.get() = 0x1234;
                std::shared_ptr<scl::CDog> shared_dog(ptr_dog);
                qInfo() << shared_dog.use_count();
                std::shared_ptr<scl::CDog> shared_dog2(ptr_dog);
                qInfo() << shared_dog.use_count();

                qInfo() << "Address: " << ptr_dog;
                qInfo() << "Address: " << shared_dog.get();
                qInfo() << "Address: " << shared_dog2.get();
                ptr_dog->bark();
                shared_dog->bark();
                shared_dog2->bark();
            }

            qInfo() << "got it...";
         };  //count = 0;

         auto foo_mixedPointersBETTER = [] ()
         {
            // STL's templated container 'shared_ptr' holds CDog;
            // shared_dog.get() = 0x1234;
            std::shared_ptr<scl::CDog> shared_dog(new scl::CDog("Tank"));
            // 1. "Tank" is created
            // 2. shared_dog is created.
            // Not exception safe.

            qInfo() << shared_dog.use_count();
            std::shared_ptr<scl::CDog> shared_dog2(shared_dog);
            qInfo() << shared_dog.use_count();

            qInfo() << "Address: " << shared_dog.get();
            qInfo() << "Address: " << shared_dog2.get();
            shared_dog->bark();
            shared_dog2->bark();


            qInfo() << "got it...";
         };  //count = 0;

         auto foo_mixedPointersMUCHBETTER = [] ()
         {
            // 'STL' templated container 'shared_ptr' holds 'CDog' class type;
            // invoke the 'STL' make_shared templated function member holding
            // 'CDog' with paramter "Tank"
            std::shared_ptr<scl::CDog> shared_dog = std::make_shared<scl::CDog>("Tank");
            // Exception safe.

            shared_dog->bark();
            (*shared_dog).bark();

            qInfo() << shared_dog.get();
            qInfo() << shared_dog.use_count();

            // static_pointer_cast;
            // dynamic_pointer_cast;
            // const_pointer_cast;

            qInfo() << "got it...";
         };  //count = 0;

        auto foo_complexPointers = []()
        {
            std::shared_ptr<scl::CDog> shared_ptr1 = std::make_shared<scl::CDog>("Gunner");
            std::shared_ptr<scl::CDog> shared_ptr2 = std::make_shared<scl::CDog>("Tanker");
            
            // Gunner is delete;
            shared_ptr1 = shared_ptr2;  
            shared_ptr1 = nullptr;
            shared_ptr1.reset();

            qInfo() << "got it...";
        };

        auto foo_customDeleter = []()
        {
            // using default deleter: operator deleter;
            std::shared_ptr<scl::CDog> shared_ptr1 = std::make_shared<scl::CDog>("Gunner");
            std::shared_ptr<scl::CDog> shared_ptr2 = std::shared_ptr<scl::CDog>(new scl::CDog("Gunner"),
                                                     [](scl::CDog * ptr_dog)
                                                     {
                                                        qInfo() << "Custome Deleter ...";
                                                        delete ptr_dog;
                                                        ptr_dog = nullptr;
                                                     });

            qInfo() << "got it...";
        };

        auto foo_returnRawPointer = []()
        {
            std::shared_ptr<scl::CDog> shared_ptr = std::make_shared<scl::CDog>("Gunner");
            
            qInfo() << "Raw Pointer: " << shared_ptr.get();
            scl::CDog * ptr_dog = shared_ptr.get();
            qInfo() << "Raw Pointer: " << ptr_dog;

            // undefined behavior;
            // delete ptr_dog;
            // ptr_dog = nullptr;

            qInfo() << "got it...";
        };

        foo_rawPointer();
        foo_sharedPointer();
        // foo_mixedPointersBADBADBAD();
        foo_mixedPointersBETTER();
        foo_mixedPointersMUCHBETTER();
        foo_complexPointers();
        foo_customDeleter();
        foo_returnRawPointer();

        qInfo() << "got it...";       

    }

    {
        std::shared_ptr<scl::CSharedDog> ptr_sharedDog1(new scl::CSharedDog("Gunner"));
        std::shared_ptr<scl::CSharedDog> ptr_sharedDog2(new scl::CSharedDog("Smookey"));

        ptr_sharedDog1 = ptr_sharedDog2;

        ptr_sharedDog2->show_friend();

        qInfo() << "got it...";       
    }

    {
        std::unique_ptr<scl::CDog> ptr_dog(new scl::CDog("Gunner"));
        ptr_dog->bark();
        (*ptr_dog).bark();

        scl::CDog * some_ptr = ptr_dog.release();

        if (!ptr_dog)
            qInfo() << "unique ptr is emtpy ...";
        
        qInfo() << "got it...";       
    }

    {
        QObject *ptr_obj = new QObject(&a);
        ptr_obj->setObjectName("My object");

        // Qt's templated class QPointer has datatype QObject;
        QPointer<QObject> ptr(ptr_obj);

        scl::CTest my_test;
        my_test.widget_ = ptr;
        my_test.use_widget();

        if (ptr.data())
            qInfo() << ptr.data();
        
        delete ptr_obj;
        my_test.use_widget();
        
        qInfo() << "got it...";       
    }

    {
        auto dostuff = []()
        {
            QScopedPointer<scl::CTest> my_pointer(new scl::CTest());

            useit( my_pointer.data());
        };

        for (int i = 0; i < 5; i++)
        {
            dostuff();
        }
        
        qInfo() << "got it...";       
    }

    return a.exec();
}
