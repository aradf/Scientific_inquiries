#ifndef FELINE2_HH                    // The compiler understand to define if it is not defined.
#define FELINE2_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CFeline2 : public QObject
{
    Q_OBJECT
public:
    CFeline2(QObject * parent = nullptr);

    void speak();
    
signals:

public slots:    
};

} // end of Scientific Computational Library.

#endif 