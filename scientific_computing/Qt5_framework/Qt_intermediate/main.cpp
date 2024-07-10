#include <QtCore/QCoreApplication>   // Gives access to QCoreApplication.
#include <QDebug>                    // Gives access to qDebug, qInfo.
#include <QProcessEnvironment>       // Gives access to QProcessEnvironment
#include <QPointer>
#include <QScopedPointer>
#include <QSharedPointer>
#include <QVector>
#include <QList>
#include <QHash>
#include <QSet>
#include <QMap>
#include <QLinkedList>
#include <QSettings>

#include <iostream>                  // Gives std::cerr

#include <../include/first.h>
#include <../include/test.h>
#include <../include/consumer.h>

// Pointer object pointes to class type 'CTest'
// ptr_obj == 0x1234 (l-value); *ptr_obj is content
// or temp data (r-value); &ptr_obj is 0xABCD
void useit(scl::CTest * ptr_obj)
{
    if (!ptr_obj)
        return;

    ptr_obj->print_name();
    qInfo() << "using " << ptr_obj;
}

void dostuff()
{
    scl::CTest * test = new scl::CTest();

    try
    {
        test->print_name();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        delete test;
        test = nullptr;
    }
    if(test)       
    {
        delete test;
        test = nullptr;
    }

    // Qt's template class holding object of class type
    // 'CTest'; my_pointer is an object of 'QScopedPointer'
    // allocating memory for 'CTest'
    QScopedPointer<scl::CTest> my_pointer(new scl::CTest());

    useit(my_pointer.data());

    // pointer is automaticlly deleted, since it is scoped.
}

QSharedPointer<scl::CTest> create_sharedPointer()
{
    scl::CTest * test = new scl::CTest();
    test->setObjectName("Test Object");

    // Use Lambda function: It is template with one parameter.
    auto lambda_delete = [](scl::CTest * ptr_test){ 
                                                   qInfo() << "Lambda Delete Later" << ptr_test; 
                                                   ptr_test->deleteLater(); 
                                                   return ;
                                                   };

    QSharedPointer<scl::CTest> ptr(test, lambda_delete);

    // create a copy.
    return ptr;
}

// The QSharedPointer class holds a strong reference to a shared pointer. More...
// Qt template class 'QSharedPointer' holds a strong reference to shared pointer.
// The holds an object of 'CConsumer' type.  object is ptr;
void do_sharedointerStuff(QSharedPointer<scl::CConsumer> ptr)
{
    qInfo() << "In Function " << ptr.data()->tester;
    ptr.clear();
}

// QSharedPointer is a Qt template holds strong refereance to an object of
// 'CConsumer' class type.
QSharedPointer<scl::CConsumer> make_consumer()
{
    // QSharedPointe is Qt template holds strong refereance to an object of 
    // 'CConsumer' class type;  object obj allocates memory for 'CConsumer'
    // class type and uses the QObject's delete later function.
    QSharedPointer<scl::CConsumer> c_ptr (new scl::CConsumer, &QObject::deleteLater);

    // QSharedPointer is Qt template holds strong referance to class type 'CTest';
    // object t_ptr is poplated by the return object of 'create_sharedPointer' function.
    QSharedPointer<scl::CTest> t_ptr = create_sharedPointer();

    c_ptr.data()->tester.swap(t_ptr);
    do_sharedointerStuff(c_ptr);

    return c_ptr;
}

// Template class holding type 'T';  function fill return void
// The paramter is a referance or address (l-value) of type 'T'
template<class T>
void fill(T &container)
{
    for (int i = 0; i < 5; i++)
        container.append(i);
}

template<class T>
void display(T &container)
{
    for(int i = 0; i < container.length(); i++)
    {
        if (i > 0)
        {
            // Key word 'auto' sets the object current to the right object type.
            // STL's container class 'reinterpret_cast' holds STL's 'uintptr_t'
            // structure; the address of the function parameter is passed as 
            // a parameter.
            auto current = reinterpret_cast<std::uintptr_t>(&container.at(i));
            auto previous = reinterpret_cast<std::uintptr_t>(&container.at(i-1));
            auto distance = current - previous;
            qInfo() << i 
                    << "At: " 
                    << current 
                    << "Prev: " 
                    << previous
                    << "Dist: " 
                    << distance;
        }
        else
            {
                qInfo() << i << &container.at(i) << "Distance: 0";
            }
    }
}

/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */
int main (int argc, char * argv[])
{
    QCoreApplication a(argc, argv);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();

    QString path = env.value("Path");
    qDebug() << "Path Values: " 
             << path;

    {
        //QSetting
        QCoreApplication::setOrganizationName("VoidRealms");
        QCoreApplication::setOrganizationDomain("voidrealms.com");
        QCoreApplication::setApplicationName("Qt_training");

        QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());
        
        // Save settings.
        settings.setValue("test",123);

        // Read the setting back.
        qInfo() << settings.value("test").toString();
        qInfo() << settings.value("test").toInt();

        qInfo() << "Done";        
    }

    {
        QSettings settings(QCoreApplication::organizationName(), QCoreApplication::applicationName());


        // Lambda Function save_age returns void; pointer object to Qt's 'QSetting'
        // setting is a pointer object 0x1234 (l-value); *settings is (r-value) 
        // the temp data or content; &setting 0xABCD; 
        auto save_age = []
                        (QSettings * setting, QString group, QString name, int age)
                        {
                            setting->beginGroup(group);
                            setting->setValue(name,age);
                            setting->endGroup();
                            return;                          
                        };

        auto get_age = []
                      (QSettings *setting, QString group, QString name)
                      {
                            int value = 0;
                            setting->beginGroup(group);

                            if(!setting->contains(name)) 
                            {
                                qWarning() << "Does not contain " << name << " in " << group;
                                setting->endGroup();
                                return 0;
                            }

                            bool ok;
                            value = setting->value(name).toInt(&ok);
                            setting->endGroup();

                            if(!ok) 
                            {
                                qWarning() << "Failed to convert ot int!!!";
                                return 0;
                            }
                            return value;                      
                       };

        save_age(&settings,"people","Hello",44);
        save_age(&settings,"beer","twoheart",1);
        save_age(&settings,"beer","Hello",11);

        qInfo() << "TwoHeart" << get_age(&settings,"beer","twoheart");
        qInfo() << "Hello (people)" << get_age(&settings,"people","Hello");
        qInfo() << "Hello (beer)" << get_age(&settings,"beer","Hello");
        qInfo() << "Good Bye ...";
    }

   {
        qInfo() << "\n\n";
        scl::CFirst * first = new scl::CFirst(&a);

        delete first;
        first = nullptr;
        qInfo() << "Good Bye ...";
   }

   {
        qInfo() << "\n\n";
        // pass a referenace to the QCoreApplication, so when the application is
        // closed this pointer object 'ptr_test' is deleted.
        // scl::CTest * ptr_test = new scl::CTest(nullptr);     // Should have a parent &a BAD BAD BAD
        scl::CTest * ptr_test = new scl::CTest(&a);

        ptr_test->setObjectName("Parent");
        for(int i = 0; i < 5 ; i++)
        {
            ptr_test->make_child(QString::number(i));
        }

        delete ptr_test;
        ptr_test = nullptr;
        qInfo() << "Good Bye ...";
   }

   {
        // QPointer
        qInfo() << "\n\n";
        // Pointer object ptr_object is a of class type 'QObject';
        // ptr_object == 0x1234 (l-value); *ptr_object = (r-value);
        // &ptr_object = 0xABCD;
        QObject * ptr_obj = new QObject(&a);
        ptr_obj->setObjectName("My Object");

        // Qt template class 'QPointer' holds QObject class type;
        QPointer<QObject> ptr_Qt(ptr_obj);

        scl::CTest test;
        test.widget_ = ptr_Qt;
        test.use_widget();

        if (ptr_Qt.data())
            qInfo() << ptr_Qt.data();

        delete ptr_obj;
        ptr_obj = nullptr;
        test.use_widget();

        qInfo() << "Good Bye ...";
   }

   {
        // QScopedPointer:  This class stores a pointer.
        // As the scoped pointer object goes out of scope, the 
        // pointer it holds is automatically deleted.
        qInfo() << "\n\n";

        for (int i = 0; i < 3; i++)
            dostuff();

        qInfo() << "Good Bye ...";
   }

   {
        // QScopedPointer:  This class stores a pointer.
        // As the scoped pointer object goes out of scope, the 
        // pointer it holds is automatically deleted.
        qInfo() << "\n\n";

        qInfo() << "Good Bye ...";
   }

    {
        // QSharedPointer:  This class stores a pointer.
        // As the scoped pointer object goes out of scope, the 
        // pointer it holds is automatically deleted.
        // it is a strong refreance to QSharedPointer.
        qInfo() << "\n\n";
        QSharedPointer<scl::CConsumer> consumer = make_consumer();
        qInfo() << "In main ..." << consumer.data()->tester;

        consumer.clear();

        qInfo() << "Good Bye ...";
    }

    {
        //QVector Qt template
        // Qt's container 'QVector' holds integers; Qt's container 
        // QList holds integers;
        QVector<int> my_vector;
        QList<int> my_list;

        fill(my_vector);
        fill(my_list);

        qInfo() << "int takes " << sizeof(int) << "bytes in RAM";
        qInfo() << "QVector and std::vector containers use contingous locations in memory.";
        display(my_vector);
        qInfo() << "QList and std::list containers uses what ever is available in memory.";
        display(my_list);

        qInfo() << "Good Bye ...";
    }

    {
        // QHash<key, T>: Qt container class that stores key/value pairs.
        QHash<QString, int> ages;
        ages.insert("Hello", 44);
        ages.insert("Bye", 45);
        ages.insert("what is up", 46);
        ages.insert("Ringo", 47);
        ages.insert("Night", 48);

        qInfo() << "Hello is: " << ages["Hello"] << "years old";
        qInfo() << "Keys: " << ages.keys() << "Size: " << ages.size();

        foreach(QString key_item, ages.keys())
            qInfo() << key_item << " : " << ages[key_item];

        qInfo() << "Good Bye ...";
    }

    {
        // QSet is Qt container class
        QSet< QString > many_people;
        many_people << "Hello" << "world" << "what" << "up";
        many_people.insert("Rango");
        
        qInfo() << "Hello is in collection " << many_people.contains("Hello");

        foreach(QString item, many_people)
            qInfo() << item ;

        qInfo() << "Good Bye ...";
    }

    {
        // QMap: This is a Qt template class that contains red-black-tree-based dictionary.
        // it stores key/value pairs and provides fast lookup.
        QMap< QString, int> ages;
        ages.insert("Hello", 44);
        ages.insert("Bye", 45);
        ages.insert("what is up", 46);
        ages.insert("Ringo", 47);
        ages.insert("Night", 48);

        qInfo() << "Hello is: " << ages["Hello"] << "years old";
        qInfo() << "Keys: " << ages.keys() << "Size: " << ages.size();

        foreach(QString key_item, ages.keys())
            qInfo() << key_item << " : " << ages[key_item];


        qInfo() << "Good Bye ...";
    }

    {
        //QLinkedlist
        QLinkedList<int> list;
        for(int i = 0; i < 10; i++) {
            list.append(i);
        }

        list.removeFirst();
        list.removeLast();
        list.removeOne(5);
        if(list.contains(3)) qInfo() << "Contains 3";
        list.clear();

        qInfo() << "Done";        
    }

    return a.exec();
}
