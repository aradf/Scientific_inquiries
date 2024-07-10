#include <QObject>
#include "../include/source.h"

scl::CSource::CSource(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CSource invoked ...";
}

scl::CSource::~CSource()
{
    qInfo() << "Destructor: CSource invoked ...";
}

void scl::CSource::test()
{
    // Make a phone call or send a signal.
    emit this->my_signal("hello world");
}
