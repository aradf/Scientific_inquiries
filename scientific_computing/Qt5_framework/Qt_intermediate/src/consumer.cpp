#include <QObject>
#include "../include/consumer.h"

scl::CConsumer::CConsumer(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CConsumer invoked ...";
}

scl::CConsumer::~CConsumer()
{
    qInfo() << this << "Destructor: CConsumer invoked ...";
}

