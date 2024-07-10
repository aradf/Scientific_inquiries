#include <QObject>
#include "../include/feline2.h"

scl::CFeline2::CFeline2(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CFeline2 invoked ...";
}

void scl::CFeline2::speak()
{
    qDebug() << "speak: CFeline2 invoked ...";
}
