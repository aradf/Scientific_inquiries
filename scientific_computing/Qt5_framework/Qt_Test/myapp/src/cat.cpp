#include "../include/cat.h"

scl::CCat::CCat(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CCat invoked ...";
}

scl::CCat::~CCat()
{
    qInfo() << this << "Destructor: " << this->some_string_ << "invoked ...";
}

void scl::CCat::test()
{
    qInfo() << "test";
}

void scl::CCat::meow()
{
    qInfo() << "meow";
}

void scl::CCat::sleep()
{
    qInfo() << "sleep";
}

void scl::CCat::speak(QString value)
{
    qInfo() << "speak:" << value;
}