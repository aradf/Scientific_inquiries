#include <QObject>
#include "../include/radio.h"

scl::CRadio::CRadio(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CRadio invoked ...";
}

scl::CRadio::~CRadio()
{
    qInfo() << "Destructor: CRadio invoked ...";
}

void scl::CRadio::listen(int channel, QString name, QString message)
{
    qInfo() << channel << " " << name << " " << message;
}
