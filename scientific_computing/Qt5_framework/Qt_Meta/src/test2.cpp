#include "../include/test2.h"

scl::CTest2::CTest2(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CTest2 invoked ...";
}

scl::CTest2::~CTest2()
{
    qInfo() << this << "Destructor: CTest2 invoked ...";
}

