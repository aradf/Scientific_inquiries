#include "../include/mylib.h"

Mylib::Mylib()
{
    qInfo() << this << "Mylib Constructor invoked ...";
}

Mylib::~Mylib()
{
    qInfo() << this << "Mylib Destructor invoked ...";
}

void Mylib::test()
{
    qInfo() << "Hello from our static lib!";
}

QString Mylib::get_string()
{
    QString some_string = "Static Library";
    return some_string;
}