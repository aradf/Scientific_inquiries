#include <QObject>
#include "../include/station.h"

scl::CStation::CStation(QObject * parent, 
                        int channel, 
                        QString name) 
                        : QObject(parent)
{
    this->channel_ = channel;
    this->name_ = name;
    qInfo() << this << "Constructor: CStation invoked ...";
}

scl::CStation::~CStation()
{
    qInfo() << "Destructor: CStation invoked ...";
}

void scl::CStation::broadcast(QString message)
{
    emit send(this->channel_, this->name_, message);
}

