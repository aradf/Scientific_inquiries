#ifndef CANINE_HH                    // The compiler understand to define if it is not defined.
#define CANINE_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()
#include <mammel.h>

// Scientific Computational Library
namespace scl
{

// QOBJECT macro works to bind all Qt libraries to tie 
// all its features to the binary.

// explicit: Mark this constructor to not implicitly
// convert types (type casting)
class CCanine : public CMammel
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
    CCanine(QObject * parent = 0);
    virtual ~CCanine();
    void speak(QString message);
    void bark() {qInfo() << "Bark ...";}

signals:

public slots:    
};

} // end of Scientific Computational Library.

#endif 