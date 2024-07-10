#ifndef CTest_HH                    // The compiler understand to define if it is not defined.
#define CTest_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()
#include <QMap>
#include <QDataStream>

#include <string.h>                  // gives access to strcpy.

// Scientific Computational Library
namespace scl
{
class CTest : public QObject
{
    Q_OBJECT
public:
    explicit CTest(QObject * parent = 0);
    virtual ~CTest();
    
    QString some_string_;
    char name_[50];

    // User defined copy construct ...
    // Parameter is an object of class type 'CTest';
    // it has passed by referance or address.
    CTest(CTest & rhs)
    {
        this->some_string_ = rhs.some_string_;
        strcpy(name_,rhs.name_);
    }

    // User defined copy assignment constructor ...
    // return the referance or address of a class type CTest;
    // over-load the '=' operator; paramter is an object of 
    // class type 'CTest' that has been passed by referance
    // or address.
    CTest& operator=(CTest& other)
    {
        this->some_string_ = other.some_string_;
        strcpy(name_, other.name_);
        return *this;
    }

    void fill();
    QString name();
    void set_name(QString value);
    QMap <QString, QString> map();

    // user define operator overload
    // this is a friend function member; return the address or by referance an
    // object of Qt Core Framework 'QDataStream'; the '<<' right shift operator
    // is over-loaded; parameters are objects of class type QDataStream passed
    // by referance or address and const object of class type 'CTest' passed
    // by referance or address.
    friend QDataStream& operator << (QDataStream &stream, const CTest &t)
    {
        qInfo() << "Operand ";
        stream << t.m_name_;
        stream << t.m_map_;

        return stream;        
    }

    // friend key world; return a referance or address of Qt's class type
    // QDataStream; over-load the right shift operator '>>'; paramters are
    // object passed by address or referance of Qt's class type QDataStram;
    // a const object passed by referance or address of class 'CTest';
    friend QDataStream& operator >>(QDataStream &stream, CTest &t)
    {
        qInfo() << "Operand ";
        stream >> t.m_name_;
        stream >> t.m_map_;

        return stream;        
    }

signals:

public slots:    

private:
    QString m_name_;
    // Qt Core Framework has a template class type QMap; it holds data type QString; 
    // an object variable
    QMap <QString, QString> m_map_;
};

} // end of Scientific Computational Library.

#endif 