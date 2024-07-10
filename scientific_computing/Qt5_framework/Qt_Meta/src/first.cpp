#include <QObject>
#include "../include/first.h"

scl::CFirst::CFirst(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CFirst invoked ...";
}

scl::CFirst::~CFirst()
{
    qInfo() << this << "Destructor: CFirst invoked ...";
}

