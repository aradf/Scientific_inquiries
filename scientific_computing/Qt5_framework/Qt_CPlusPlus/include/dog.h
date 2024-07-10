#ifndef DOG_HH                    // The compiler understand to define if it is not defined.
#define DOG_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

#include <string.h>                  // gives access to std::strcpy.

// Scientific Computational Library
namespace scl
{
class CDog : public QObject
{
    Q_OBJECT
public:
    explicit CDog(QObject * parent = 0);
    CDog(std::string name);
    CDog();
    virtual ~CDog();

    // User defined copy construct ...
    CDog(CDog & rhs)
    {
        this->name_string_ = rhs.name_string_;
        strcpy(name_,rhs.name_);
    }

    // User defined copy assignment constructor ...
    CDog& operator=(CDog& other)
    {
        this->name_string_ = other.name_string_;
        strcpy(name_, other.name_);
        return *this;
    }
    void bark() 
    {
        qInfo() << "CDog Barked: " << QString::fromUtf8(name_string_.c_str()) ;
    }

signals:

public slots:    

private:
    std::string name_string_;
    char name_[50];

};

} // end of Scientific Computational Library.

#endif 