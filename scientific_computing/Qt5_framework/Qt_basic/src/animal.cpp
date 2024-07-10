#include <QObject>
#include "../include/animal.h"

scl::CAnimal::CAnimal(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CAnimal invoked ...";
}

scl::CAnimal::~CAnimal()
{
    qInfo() << "Destructor: CAnimal invoked ...";
}

void scl::CAnimal::speak(QString message)
{
    qDebug() << message;
}

scl::CAnimal2::CAnimal2(QObject * parent, QString name) : QObject(parent)
{
    QString name_onStack = name;
    this->name_ = name;
    qInfo() << "Parent address: " << parent;
    qInfo() << this << "Constructor: CAnimal2 invoked ..." << this->name_;

    qInfo() << "Animal2 Name Parameter (Stack)" << &name << " " << name;
    qInfo() << "Animal2 Name Parameter (Stack)" << &name_onStack << " " << name_onStack;
    qInfo() << "Animal2 Name_ Parameter (Not Stack)" << &name_ << " " << name_;
}

scl::CAnimal2::~CAnimal2()
{

}

void scl::CAnimal2::say_hello(QString message)
{
    qInfo() << "Animal2 Name_ Parameter (Not Stack)" << &name_ << " " << name_;
}
