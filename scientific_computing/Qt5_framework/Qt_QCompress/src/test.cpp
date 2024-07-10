#include <QObject>
#include "../include/test.h"

scl::CTest::CTest(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CTest invoked ...";
}

scl::CTest::~CTest()
{
    qInfo() << this << "Destructor: CTest invoked ...";
}



void scl::CTest::fill()
{
    this->m_name_ = "test object";
    for (int i = 0; i < 10; i++)
    {
        QString num = QString::number(i);
        QString key = "key: " + num;
        QString value = "This is item number: " + num;
        m_map_.insert(key,value);
    }

}

QString scl::CTest::name()
{
    return this->m_name_;
}

void scl::CTest::set_name(QString value)
{
    this->m_name_ = value;
}

// return by value an object of QMap; QMap is a 
// tempalte class in Qt Core framework.  The 
// class type is QString.
QMap<QString, QString> scl::CTest::map()
{
    return this->m_map_;
}

