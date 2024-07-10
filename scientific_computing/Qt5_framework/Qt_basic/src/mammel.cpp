#include <QObject>
#include "../include/mammel.h"

scl::CMammel::CMammel(QObject * parent) : CAnimal(parent)
{
    qInfo() << this << "Constructor: CAnimal invoked ...";
}

scl::CMammel::~CMammel()
{
    qInfo() << "Destructor: CAnimal invoked ...";
}

void scl::CMammel::speak(QString message)
{
    qDebug() << message;
}
