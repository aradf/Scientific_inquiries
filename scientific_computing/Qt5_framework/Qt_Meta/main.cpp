#include <QCoreApplication>
#include <QTimer>
#include <QDebug>
#include <QMetaObject>
#include <QMetaMethod>
#include <QMetaProperty>
#include <QMetaClassInfo>
#include <QVariant>

#include "../include/test.h"
#include "../include/watcher.h"
#include "../include/test1.h"
#include "../include/test2.h"

void test()
{
    qInfo() << "Thank you for waiting ...";
}

void list_properties(QObject * ptr_obj)
{
    int i = 0;
    // 'metaObject' return a pointer to the metaObject of QObject;
    // 'methodCount' is a function member of QmetaObject
    for (i = 0; i < ptr_obj->metaObject()->propertyCount(); i++)
    {
        QMetaProperty p = ptr_obj->metaObject()->property(i);
        qInfo() << "properties: " << p.name() << p.typeName();
    }
}


void list_methods(QObject * ptr_obj)
{
    int i = 0;
    // 'metaObject' return a pointer to the metaObject of QObject;
    // 'methodCount' is a function member of QmetaObject
    for (i = 0; i < ptr_obj->metaObject()->methodCount(); i++)
    {
        QMetaMethod m = ptr_obj->metaObject()->method(i);
        qInfo() << "Method: " << m.methodSignature();
    }
}

void test_property(QObject *parent, QString name, QVariant value)
{
    foreach(QObject *child_item, parent->children())
    {
        child_item->setProperty(name.toLatin1(), value);
    }
}

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
        qInfo() << "Please wait ...";
        QTimer::singleShot(1000, &test);

        qInfo() << "Life is good ...";
    }

//    {
        scl::CTest my_test;
        my_test.do_stuff();
        qInfo() << "Life is good ...";
//    }

//    {
        CWatcher watcher;
        qInfo() << "Life is good ...";
//    }

    {
        // Lambda Function: return a void; parameter: pointer object 
        // to some memory location holding class type QObject; 
        // ptr_obj == 0x1234 (l-value); *ptr_obj is the r-value, 
        // temp-data, content of memory location;  arrow operator '->'
        // will invoke the methods of class type QObject.
        auto list_children = [](QObject *ptr_obj)
        {
            qInfo() << "QObject: " << ptr_obj;
            foreach(QObject *ptr_child, ptr_obj->children())
            {
                qInfo() << ptr_child;
                list_methods(ptr_child);
                list_properties(ptr_child);
                qInfo() << "\n";
            }
        };
        // object variable 'parent' is Qt's class type 'QObject'; it holds
        // a r-value; ptr_timer is a pointer object; pointing to a 
        // memory block; holding QTimer class type; ptr_timer == 0x1234 (l-value);
        // *ptr_timer is r-value, temp data or content;
        QObject parent;

        // note &parent == 0x1234;
        QTimer *ptr_timer =  new QTimer(&parent);

        // Make some children;  The children are under the memory hierachy of
        // &parent as it is specifed in 'scl::CTest1(&parent);  The &parent is 
        // memory block;  heirachy tree of other objects;
        for (int i = 0; i < 5; i ++)
        {
            // 'test1' is a pointer object; it points to some memory location
            // with content 'CTest1'; teat1 == 0x1234 (l-value); *test is the
            // content of the memory block; test-> will invoke methods of
            // class type 'CTest1' &test1 == 0xABCD; all the pointer objects 
            // are created uner the heiarchy tree of parent;
            scl::CTest1 * test1_child = new scl::CTest1(&parent);
            test1_child->setObjectName("Child: " + QString::number(i));
        }

        test_property(&parent, "interval", 3000);
        list_children(&parent);

        // c is a pointer object; pointing to some memory location; c = 0x1234;
        // *c is the (r-value); temp data; '->' operator invokes the QObject's 
        // methods; 'parent.children()' returns a list of child objects;
        // typedef QList<QObject *> QObjectList; QList is a Qt template class
        // with datatype QObject pointers;
        foreach(QObject *c, parent.children())
        {
            // Print the address of the pointer object;  '->' operator
            // invokes the 'metaObject()' method which returns a pointer
            // to the metha object; className should return 'QObject'
            qInfo() << c;
            qInfo() << c->metaObject()->className();
            qInfo() << "Inherit: " << c->inherits("scl::CTest1");

            // methodCount returns the number of methods in the class.
            for(int m = 0; m < c->metaObject()->methodCount(); m++)
            {
                qInfo() << "Method: " << c->metaObject()->method(m).methodSignature();
            }
            qInfo() << "\n";
        }
    }

    {
        auto list_classInfo = [] (QObject *ptr_obj)
        {
            qInfo() << ptr_obj->metaObject()->className();
            for(int i = 0; i < ptr_obj->metaObject()->classInfoCount(); i++)
            {
                QMetaClassInfo c = ptr_obj->metaObject()->classInfo(i);
                qInfo() << "Property: " << c.name() << " " << c.value();
            }
        };

        scl::CTest2 test2;
        list_classInfo(&test2);

    }

    {
        auto pointer_test = [] (QObject *ptr_obj)
        {
            qInfo() << ptr_obj;
        };

        auto referance_test = [] (QObject &ptr_obj)
        {
            qInfo() << &ptr_obj;
        };

        auto copy_test = [] (QObject obj)
        {
            qInfo() << &obj;
        };

        scl::CTest2 test2;
        pointer_test(&test2);
        referance_test(test2);
        // Will not compile 'error: use of deleted function ‘QObject::QObject(const QObject&)’'
        // The macro Q_DISABLE_COPY, the compiler will delete the copy cnstructor;
        // copy_test(test2);          

    }

    return a.exec();
}
