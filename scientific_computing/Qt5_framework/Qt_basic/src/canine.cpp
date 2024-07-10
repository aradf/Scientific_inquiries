#include <QObject>
#include "../include/canine.h"

scl::CCanine::CCanine(QObject * parent) : CMammel(parent)
{
    qInfo() << this << "Constructor: CCanine invoked ...";
}

scl::CCanine::~CCanine()
{
    qInfo() << "Destructor: CCanine invoked ...";
}

void scl::CCanine::speak(QString message)
{
    qDebug() << message;
}
