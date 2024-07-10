#ifndef CONSUMER_HH                    // The compiler understand to define if it is not defined.
#define CONSUMER_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()
#include <QSharedPointer>            // gives access to QSharedPointer

#include <string.h>                  // gives access to strcpy.
#include <test.h>

// Scientific Computational Library
namespace scl
{
class CConsumer : public QObject
{
    Q_OBJECT
public:
    explicit CConsumer(QObject * parent = 0);
    virtual ~CConsumer();
    
    QString some_string_;
    char name_[50];
    // Qt tample class 'QSharedPointer' holds a 'CTest'
    // object;  It holds a strong referance to shared
    // pointer.
    QSharedPointer<scl::CTest> tester;

    // User defined copy construct ...
    CConsumer(CConsumer & rhs)
    {
        this->some_string_ = rhs.some_string_;
        strcpy(name_,rhs.name_);
    }

    // User defined copy assignment constructor ...
    CConsumer& operator=(CConsumer& other)
    {
        this->some_string_ = other.some_string_;
        strcpy(name_, other.name_);
        return *this;
    }

signals:

public slots:    
};

} // end of Scientific Computational Library.

#endif 