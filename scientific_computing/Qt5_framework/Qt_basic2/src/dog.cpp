#include <QDebug>                    // gives access to qInfo()
#include "../include/dog.h"

scl::CDog::CDog()
{
    qInfo() << this << "Constructor: CDog invoked ...";
}

scl::CDog::~CDog()
{
    qInfo() << "Destructor: CDog invoked ...";
}

