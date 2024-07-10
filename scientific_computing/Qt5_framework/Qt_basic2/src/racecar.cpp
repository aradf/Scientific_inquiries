#include <QObject>
#include "../include/racecar.h"

// pointer object of class tyep QObject: parent is 0x1234 (l-value); 
// *parent is temperary content (r-value); &parent is 0xABCD on (Stack)
scl::CRacecar::CRacecar(QObject * parent) : CCar(parent)
{
    this->supper_charger_ = true;
    this->color_ = "red";
    qInfo() << this << "Constructor: CRacecar invoked ...";
}

scl::CRacecar::~CRacecar()
{
    qInfo() << "Destructor: CRacecar invoked ...";
}

void scl::CRacecar::go_fast()
{
    qInfo() << "ZOOOOOOOOOM";
}
