#ifndef TEST2_HH                    // The compiler understand to define if it is not defined.
#define TEST2_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

#include <string.h>                  // gives access to strcpy.

// Scientific Computational Library
namespace scl
{
class CTest2 : public QObject
{
    Q_OBJECT
    Q_CLASSINFO("Author", "Yonder");
    Q_CLASSINFO("Url","https://www.yonder.com");
    Q_DISABLE_COPY(CTest2)
    // Added meta data for CTest2 class.    
public:
    explicit CTest2(QObject * parent = 0);
    virtual ~CTest2();
    
    QString some_string_;
    char name_[50];

    // User defined copy construct ...
    CTest2(CTest2 & rhs)
    {
        this->some_string_ = rhs.some_string_;
        strcpy(name_,rhs.name_);
    }

    // User defined copy assignment constructor ...
    CTest2& operator=(CTest2& other)
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