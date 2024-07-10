#include <QCoreApplication>
#include <QDebug>

#include <QMetaEnum>
#include "car.h"

//Factory method
/*
    A framework needs to standardize the architectural model for a range of applications,
    but allow for individual applications to define their own domain objects and provide for their instantiation.
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
        QMetaEnum metaEnum = QMetaEnum::fromType<CCar::Model>();
        for(int i = 0; i < metaEnum.keyCount(); i++) 
        {
            QString key = metaEnum.key(i);
            qInfo() << "Attempting to create: " << key;
            CCar::Model model = static_cast<CCar::Model>(metaEnum.keysToValue(key.toLatin1()));
            CCar *ptr_car = CCar::make(model);
            ptr_car->drive();
            qInfo() << "\n";
        }
    }

    return a.exec();
}
