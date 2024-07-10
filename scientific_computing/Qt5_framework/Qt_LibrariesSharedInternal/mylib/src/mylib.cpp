#include "../include/mylib.h"

Mylib::Mylib()
{
    qInfo() << this << "Mylib Constructor Invoked ...";
}

Mylib::~Mylib()
{
    qInfo() << this << "Mylib Destructor invoked ..."; 
}
void Mylib::test_library()
{
    qInfo() << "This is a test from shared library: mylib!";
}

QString Mylib::get_name()
{
    QString lib_name = "Mylib";
    return lib_name;
}