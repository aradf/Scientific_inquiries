#ifndef FIRST_HH                    // The compiler understand to define if it is not defined.
#define FIRST_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

#include <string.h>                  // gives access to strcpy.

// Scientific Computational Library
namespace scl
{
class CFirst : public QObject
{
    Q_OBJECT
public:
    explicit CFirst(QObject * parent = 0);
    virtual ~CFirst();
    
    // User defined copy construct ...
    CFirst(CFirst & rhs)
    {
        this->some_string_ = rhs.some_string_;
        strcpy(name_,rhs.name_);
    }

    // User defined copy assignment constructor ...
    CFirst& operator=(CFirst& other)
    {
        this->some_string_ = other.some_string_;
        strcpy(name_, other.name_);
        return *this;
    }

signals:

public slots:    

private:
    QString some_string_;
    char name_[50];

};

} // end of Scientific Computational Library.

#endif 