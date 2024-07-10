#ifndef ANIMAL_HH                    // The compiler understand to define if it is not defined.
#define ANIMAL_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CAnimal : public QObject
{
    Q_OBJECT
public:
    explicit CAnimal(QObject * parent = 0);
    virtual ~CAnimal();
    
signals:

public slots:    
};


} // end of Scientific Computational Library.

#endif 