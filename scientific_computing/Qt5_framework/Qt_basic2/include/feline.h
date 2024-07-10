#ifndef FELINE_HH                    // The compiler understand to define if it is not defined.
#define FELINE_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CFeline : public QObject
{
    Q_OBJECT
public:
    explicit CFeline(QObject * parent = 0);
    virtual ~CFeline();

    void meow();
    void hiss();
    
signals:

public slots:    
};


} // end of Scientific Computational Library.

#endif 