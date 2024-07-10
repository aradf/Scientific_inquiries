#ifndef DESTINATION_HH                    // The compiler understand to define if it is not defined.
#define DESTINATION_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CDestination : public QObject
{
    Q_OBJECT
public:
    explicit CDestination(QObject * parent = 0);
    virtual ~CDestination();
    
signals:

public slots:
    void on_message(QString message);
};

} // end of Scientific Computational Library.

#endif 