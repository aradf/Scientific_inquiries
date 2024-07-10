#include <QObject>
#include "../include/applicance.h"

scl::CApplicance::CApplicance(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CAplicance invoked ...";
}

bool scl::CApplicance::cook()
{
    return true;
}

bool scl::CApplicance::grills()
{
    return true;
}

bool scl::CApplicance::freeze()
{
    return true;
}
