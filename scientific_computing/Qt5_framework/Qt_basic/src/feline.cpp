#include <QObject>
#include "../include/feline.h"

scl::CFeline::CFeline(QObject * parent) : CMammel(parent)
{
    qInfo() << this << "Constructor: CFeline invoked ...";
}

scl::CFeline::~CFeline()
{
    qInfo() << "Destructor: CFeline invoked ...";
}

void scl::CFeline::speak(QString message)
{
    qDebug() << message;
}
