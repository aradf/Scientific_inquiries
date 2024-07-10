#ifndef RADIO_HH                    // The compiler understand to define if it is not defined.
#define RADIO_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CRadio : public QObject
{
    Q_OBJECT
public:
    explicit CRadio(QObject * parent = 0);
    virtual ~CRadio();
    
signals:
    void quit();

public slots:
    void listen(int channel, QString name, QString message);
};

} // end of Scientific Computational Library.

#endif 