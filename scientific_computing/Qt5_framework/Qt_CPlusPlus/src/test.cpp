#include <QObject>

#include "../include/test.h"

scl::CTest::CTest(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CTest invoked ...";
}

scl::CTest::~CTest()
{
    qInfo() << this << "Destructor: CTest invoked ...";
}

void scl::CTest::make_child(QString name)
{
    qInfo() << this << "make_child " << name;
    // Pointer object 'child' of class type 'CTest';
    // child == 0x1234; *child points to content of
    // class type CTest; &child is address 0xABCD

    CTest * child = new CTest(this);
    child->setObjectName(name);

}

void scl::CTest::use_widget()
{
    // check the pointer is valid;
    // widget_.data(0 return a raw pointer)
    if (!widget_.data())
    {
        qInfo() << "Invalid Pointer ...";
        return;
    }
    
    qInfo() << "Valid Poninter ... " << widget_.data();
    (widget_.data())->setObjectName("Used Widget");
}

void scl::CTest::print_name()
{
    qInfo() << this << " char " << name_;
}