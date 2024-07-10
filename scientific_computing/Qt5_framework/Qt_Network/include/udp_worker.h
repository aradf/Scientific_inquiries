#ifndef UDP_WORKER_HH                    // The compiler understand to define if it is not defined.
#define UDP_WORKER_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()
#include <QTimer>
#include <QDateTime>
#include <QUdpSocket>
#include <QNetworkDatagram>

#include <string.h>                  // gives access to strcpy.

// Scientific Computational Library
namespace scl
{
class CUdpWorker : public QObject
{
    Q_OBJECT
public:
    explicit CUdpWorker(QObject * parent = 0);
    virtual ~CUdpWorker();
    
    // User defined copy construct ...
    CUdpWorker(CUdpWorker & rhs)
    {
        this->some_string_ = rhs.some_string_;
        strcpy(name_,rhs.name_);
    }

    // User defined copy assignment constructor ...
    CUdpWorker& operator=(CUdpWorker& other)
    {
        this->some_string_ = other.some_string_;
        strcpy(name_, other.name_);
        return *this;
    }

signals:

public slots:    
    // Consume a signal that is emitted from somewhere
    void start();
    void stop();
    void timeout();
    void readyRead();
    void broadcast();

private:
    QString some_string_;
    char name_[50];
    QUdpSocket udp_socket;
    QTimer udp_timer;
    quint16 port = 2020;         // Note < 1024 those admin ports / speical services, HTTP, FTP, POP3, SMTP

};

} // end of Scientific Computational Library.

#endif 