#include <QObject>
#include "../include/laptop.h"

scl::CLaptop::CLaptop(QObject * parent) : QObject(parent)
{
    // Pointer to thee current instance of this object is 'this'.
    // it is created in the construtor.
    this->name_ = "Undefined";
    qInfo() << "CLaptop Constructor invoked ... ";
}

scl::CLaptop::CLaptop(QObject * parent, QString some_name) : QObject(parent)
{
    this->name_ = some_name;
    qInfo() << "CLaptop Constructor invoked ... " 
            << this 
            << " " 
            << this->name_;
}

scl::CLaptop::~CLaptop()
{
    qInfo() << "CLaptop Destructor invoked ... " << this->name_;
}

double scl::CLaptop::as_kilograms()
{
    return this->weight_ * 0.453592;
}

void scl::CLaptop::test()
{
    qInfo() << "testing: " << this->name_;
}
