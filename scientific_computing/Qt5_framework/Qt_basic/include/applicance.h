#ifndef APLICANCE_HH                    // The compiler understand to define if it is not defined.
#define APLICANCE_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

#include "../include/freezer.h"
#include "../include/microwave.h"
#include "../include/toaster.h"

// Scientific Computational Library
namespace scl
{

// this is not true inheritance, CFreezer, CToaster, CMicrowave are interfaces.
class CApplicance : public QObject, public CFreezer, public CToaster, public CMicrowave 
{
    Q_OBJECT
public:
    explicit CApplicance(QObject * parent = 0);
    
signals:

public: 
    bool cook();

public:
    bool grills();

public:
    bool freeze();

public slots:    
};

} // end of Scientific Computational Library.

#endif 