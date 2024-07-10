#ifndef CAR_HH                    // The compiler understand to define if it is not defined.
#define CAR_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CCar : public QObject
{
    Q_OBJECT
public:
    explicit CCar(QObject * parent = 0);
    virtual ~CCar();

    // user define copy constructor;
    CCar(QObject * parent, CCar& other) : QObject(parent)
    {
        this->color_ = other.color_;
        this->tires_ = other.tires_;
    }

    // user defined copy asignment operator
    // return the address or referance of CCar class type object;
    // operator key word for '='; function members parameter is 
    // referance or addres of clas type CCar
    CCar& operator=(CCar & other)
    {
        this->color_ = other.color_;
        this->tires_ = other.tires_;
        return *this;
    }

    QString color_ = "white";
    int tires_ = 4;
    void drive();
    void stop();

signals:

public slots:    
};


} // end of Scientific Computational Library.

#endif 