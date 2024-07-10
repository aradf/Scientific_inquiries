#include "../include/dog.h"

scl::CDog::CDog(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CDog invoked ...";
}

scl::CDog::~CDog()
{
    qInfo() << this << "Destructor: CDog invoked ..." << QString::fromUtf8(name_string_.c_str());
}

scl::CDog::CDog(std::string name)
{
    qInfo() << this << " CDog is created: " << QString::fromUtf8(name.c_str());
    this->name_string_ = name;
}
scl::CDog::CDog()
{
    qInfo() << this << "CDog is created: 'Nameless'";
}
