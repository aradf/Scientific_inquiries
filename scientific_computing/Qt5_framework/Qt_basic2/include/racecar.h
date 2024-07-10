#ifndef RACECAR_HH                    // The compiler understand to define if it is not defined.
#define RACECAR_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // Gives access to qInfo()

#include <car.h>                     // Gives access to CCar class.

// Scientific Computational Library
namespace scl
{

class CRacecar : public CCar
{
    Q_OBJECT
public:
    explicit CRacecar(QObject * parent = 0);
    virtual ~CRacecar();
    
    bool supper_charger_ = true;
    void go_fast();
signals:

public slots:    
};


} // end of Scientific Computational Library.

#endif 