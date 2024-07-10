#include <QCoreApplication>
#include <QDebug>

#include <QMetaObject>
#include <QMetaEnum>
#include "person.h"

/*
 * Builder design pattern
 * Build things in a uniform way - millions of ways to do this.
 */

#include <../include/first.h>

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

        QObject classroom;
        QMetaEnum meta_enum = QMetaEnum::fromType<CPerson::persontype>();

        for(int i = 0; i < 10; i++) 
        {
            CPerson::persontype type = CPerson::persontype::STUDENT;
            if(i == 0) 
                type = CPerson::persontype::PRINCIPAL;
            if(i == 1) 
                type = CPerson::persontype::TEACHER;

            CPerson *p = CPerson::build(type);
            p->setParent(&classroom);
        }

        // for each pointer object 'child' of class type QObject;
        // ptr_child == 0x1234; '->' arrow operator invokes a function member
        // from QObject class type.
        foreach(QObject * ptr_child, classroom.children()) 
        {
            //Qt's qobject_cast templated class holds pointer objects to 
            // CPerson class type; parameter is a pointer object to the 
            // QObject's class type;
            CPerson *ptr_person = qobject_cast<CPerson*>(ptr_child);
            if(ptr_person) 
                qInfo() << ptr_person << " is a " << meta_enum.valueToKey(ptr_person->role());
        }
    }

    return a.exec();
}
