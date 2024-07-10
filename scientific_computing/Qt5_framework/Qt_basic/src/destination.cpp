#include <QObject>
#include "../include/destination.h"

scl::CDestination::CDestination(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CDestination invoked ...";
}

scl::CDestination::~CDestination()
{
    qInfo() << "Destructor: CDestination invoked ...";
}

void scl::CDestination::on_message(QString message)
{
    qInfo() << "on_message " << message;
}