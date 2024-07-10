#include <QObject>
#include "../include/animal.h"

scl::CAnimal::CAnimal(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CAnimal invoked ...";
}

scl::CAnimal::~CAnimal()
{
    qInfo() << "Destructor: CAnimal invoked ...";
}

