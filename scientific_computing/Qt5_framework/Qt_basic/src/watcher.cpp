#include <QObject>
#include "../include/watcher.h"

scl::CWatcher::CWatcher(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CWatcher invoked ...";
}

scl::CWatcher::~CWatcher()
{
    qInfo() << "Destructor: CAnimal invoked ...";
}

QString scl::CWatcher::return_message()
{
    return this->message_;
}

void scl::CWatcher::set_message(QString some_value)
{
    this->message_ = some_value;
}


void scl::CWatcher::message_changed(QString message)
{
    qInfo() << message;
}