#include "../include/dog.h"

scl::CDog::CDog(QObject *parent) : QObject(parent)
{
    qInfo() << "Constructor";
}

void scl::CDog::initTestCase()
{
    qInfo() << "initTestCase";
}

void scl::CDog::init()
{
    qInfo() << "init";
}

void scl::CDog::cleanup()
{
    qInfo() << "cleanup";
}

void scl::CDog::cleanupTestCase()
{
    qInfo() << "cleanupTestCase";
}

void scl::CDog::bark()
{
    qInfo() << "bark bark bark";
}

void scl::CDog::rollover()
{
    qInfo() << "*rolls*";
}