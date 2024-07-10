#ifndef ANIMAL_HH                    // The compiler understand to define if it is not defined.
#define ANIMAL_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

// QOBJECT macro works to bind all Qt libraries to tie 
// all its features to the binary.

// explicit: Mark this constructor to not implicitly
// convert types (type casting)
class CAnimal : public QObject
{
    /* 
        Caused linker issue: 
            Undefined referance to virtual table or 'vtable' for scl::CAnimal::CAnimal(QObject*)
        Solution:
            Remove 'Q_OBJECT' 
            CMakeLists.txt must have 'set(CMAKE_AUTOMOC on)'
     */
    Q_OBJECT
public:
    explicit CAnimal(QObject * parent = 0);
    virtual ~CAnimal();
    void speak(QString message);
    bool is_alive() {return true;};
    
signals:

public slots:    
};

class CAnimal2 : public QObject
{
    /* 
        Caused linker issue: 
            Undefined referance to virtual table or 'vtable' for scl::CAnimal::CAnimal(QObject*)
        Solution:
            Remove 'Q_OBJECT' 
            CMakeLists.txt must have 'set(CMAKE_AUTOMOC on)'
     */
    Q_OBJECT
public:
    explicit CAnimal2(QObject * parent = 0, QString name = "");
    virtual ~CAnimal2();
    
    QString name_;
    void say_hello(QString message);
signals:

public slots:    
};

} // end of Scientific Computational Library.

#endif 