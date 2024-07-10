#ifndef SERIAL_PORT_HH                    // The compiler understand to define if it is not defined.
#define SERIAL_PORT_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

class QSerialPort;
class QSerialPortInfo;

class CSerialPort : public QObject
{
    Q_OBJECT
public:
    explicit CSerialPort(QObject * parent = 0);
    virtual ~CSerialPort();

    bool connect(QString port_name);
    qint64 write(QByteArray data);
    void serialport_info();
    
signals:
    void data_recieved(QByteArray b);

private slots:    
    void read_serialData();

private:
    QSerialPort *_serial_port;

};

#endif 