#ifndef LION_HH                    // The compiler understand to define if it is not defined.
#define LION_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

#include "feline2.h"

// Scientific Computational Library
namespace scl
{

class CLion : public CFeline2
{
    Q_OBJECT
public:
    explicit CLion(QObject * parent = 0);
    void speak();
    
signals:

public slots:    
};

} // end of Scientific Computational Library.

#endif 