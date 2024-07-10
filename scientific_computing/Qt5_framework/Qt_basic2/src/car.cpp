#include <QObject>
#include "../include/car.h"

// default constructor.
scl::CCar::CCar(QObject * parent) : QObject(parent)
{
    QString color_ = "white";
    int tires_ = 4;

    qInfo() << this << "Constructor: CCar invoked ...";
}

scl::CCar::~CCar()
{
    qInfo() << "Destructor: CCar invoked ...";
}

void scl::CCar::drive()
{
    qInfo() << "Drive";
}
void scl::CCar::stop()
{
    qInfo() << "Stop";
}
