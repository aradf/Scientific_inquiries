#include <QObject>
#include "../include/feline.h"

scl::CFeline::CFeline(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CFeline invoked ...";
}

scl::CFeline::~CFeline()
{
    qInfo() << "Destructor: CFeline invoked ...";
}

void scl::CFeline::meow()
{

}
void scl::CFeline::hiss()
{
    
}
