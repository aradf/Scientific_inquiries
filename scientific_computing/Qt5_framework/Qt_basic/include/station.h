#ifndef STATION_HH                    // The compiler understand to define if it is not defined.
#define STATION_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CStation : public QObject
{
    Q_OBJECT
public:
    explicit CStation(QObject * parent = 0, int channel = 0, QString name = "unkown");
    virtual ~CStation();
    
    QString name_;
    int channel_;
signals:
    void send(int channel, QString name, QString message);

public slots:
    void broadcast(QString message);
};

} // end of Scientific Computational Library.

#endif 