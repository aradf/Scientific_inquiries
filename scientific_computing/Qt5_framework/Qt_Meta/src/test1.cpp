#include <QObject>
#include "../include/test1.h"

scl::CTest1::CTest1(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CTest1 invoked ...";
}

scl::CTest1::~CTest1()
{
    qInfo() << this << "Destructor: CTest1 invoked ...";
}

void scl::CTest1::do_stuff()
{

}

void scl::CTest1::do_stuff(QString param)
{
    Q_UNUSED(param);
}

void scl::CTest1::my_slot()
{

}
