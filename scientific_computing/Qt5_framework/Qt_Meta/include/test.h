#ifndef TEST_HH                    // The compiler understand to define if it is not defined.
#define TEST_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()
#include <QTime>
#include <QTimer>

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
    CTest(CTest & rhs)
    {
        this->some_string_ = rhs.some_string_;
        strcpy(name_,rhs.name_);
    }

    // User defined copy assignment constructor ...
    CTest& operator=(CTest& other)
    {
        this->some_string_ = other.some_string_;
        strcpy(name_, other.name_);
        return *this;
    }

signals:

public slots:
    //Consume signals;
    void timeout();
    void do_stuff();    

private:
    QTimer timer_;
    int number_;

};

} // end of Scientific Computational Library.

#endif 