#include <QObject>
#include "../include/test_watcher.h"

scl::CTestwatcher::CTestwatcher(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CTestwatcher invoked ...";
}

scl::CTestwatcher::~CTestwatcher()
{
    qInfo() << "Destructor: CTestwatcher invoked ...";
}

QString scl::CTestwatcher::message()
{
    return this->message_;
}

void scl::CTestwatcher::set_message(QString value)
{
    this->message_ = value;
    emit message_changed(this->message_);
}

