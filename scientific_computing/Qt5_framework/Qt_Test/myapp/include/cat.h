#ifndef CAT_HH                    // The compiler understand to define if it is not defined.
#define CAT_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

#include <string.h>                  // gives access to strcpy.

// Scientific Computational Library
namespace scl
{
class CCat : public QObject
{
    Q_OBJECT
public:
    explicit CCat(QObject * parent = 0);
    virtual ~CCat();
    
    // User defined copy construct ...
    CCat(CCat & rhs)
    {
        this->some_string_ = rhs.some_string_;
        strcpy(name_,rhs.name_);
    }

    // User defined copy assignment constructor ...
    CCat& operator=(CCat& other)
    {
        this->some_string_ = other.some_string_;
        strcpy(name_, other.name_);
        return *this;
    }
    void set_someString(QString some_string)
    {
        some_string_ = some_string;
    }

signals:

public slots:    
    void test();

private slots:
    void meow();
    void sleep();
    void speak(QString value);
    
private:
    QString some_string_;
    char name_[50];

};

} // end of Scientific Computational Library.

#endif 