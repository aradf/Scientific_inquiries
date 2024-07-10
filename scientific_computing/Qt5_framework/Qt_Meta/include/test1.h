#ifndef TEST1_HH                    // The compiler understand to define if it is not defined.
#define TEST1_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

#include <string.h>                  // gives access to strcpy.

// Scientific Computational Library
namespace scl
{
class CTest1 : public QObject
{
    Q_OBJECT
public:
    explicit CTest1(QObject * parent = 0);
    virtual ~CTest1();
    
    QString some_string_;
    char name_[50];

    // User defined copy construct ...
    CTest1(CTest1 & rhs)
    {
        this->some_string_ = rhs.some_string_;
        strcpy(name_,rhs.name_);
    }

    // User defined copy assignment constructor ...
    CTest1& operator=(CTest1& other)
    {
        this->some_string_ = other.some_string_;
        strcpy(name_, other.name_);
        return *this;
    }

    void do_stuff();
    void do_stuff(QString param);

signals:
    void my_signal();

public slots:    
    void my_slot();
};

} // end of Scientific Computational Library.

#endif 