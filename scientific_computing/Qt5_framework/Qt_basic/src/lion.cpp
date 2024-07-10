#include <QObject>
#include "../include/lion.h"

scl::CLion::CLion(QObject * parent) : CFeline2(parent)
{
    qInfo() << this << "Constructor: CLine invoked ...";
}

void scl::CLion::speak()
{
    qInfo() << "speak: CLine invoked ...";   

    // call the function from the base
    scl::CFeline2::speak(); 
}