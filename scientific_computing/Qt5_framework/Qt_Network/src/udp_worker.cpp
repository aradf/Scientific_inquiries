#include <QObject>
#include "../include/udp_worker.h"

scl::CUdpWorker::CUdpWorker(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CUdpWorker invoked ...";
    // connect signal QTimer::timeout (udp_timer) with a consuming slot CUdpWorker::timeout (this)
    // connect signal QUdpSocket::readyRead (udp_socket) with a consuming slot CUdpWorker::readyRead
    connect(&udp_timer, &QTimer::timeout, this, &scl::CUdpWorker::timeout);
    connect(&udp_socket, &QUdpSocket::readyRead, this, &scl::CUdpWorker::readyRead); 
    udp_timer.setInterval(1000);
}

scl::CUdpWorker::~CUdpWorker()
{
    qInfo() << this << "Destructor: CUdpWorker invoked ...";
}

void scl::CUdpWorker::start()
{
    if(!udp_socket.bind(port))
    {
        qInfo() << udp_socket.errorString();
        return;
    }
    qInfo() << "Started UDP on " << udp_socket.localAddress() << " : " << udp_socket.localPort();
    broadcast();
}
void scl::CUdpWorker::stop()
{
    udp_timer.stop();
    udp_socket.close();
    qInfo() << "Stopped ...";
}

void scl::CUdpWorker::timeout()
{
    QString date = QDateTime::currentDateTime().toString();
    QByteArray data = date.toLatin1();

    // Instantiate a Network Datagram protocal (C structure) with the 'data' content 
    // and broad cast it on (broadcast range) using user-defined port.
    QNetworkDatagram datagram(data, QHostAddress::Broadcast, port);
    qInfo() << "Send: " << data;

    // Use the C structure datagram and write it to the upd_socket.
    udp_socket.writeDatagram(datagram);
}

void scl::CUdpWorker::readyRead()
{
    // Read incoming information:  When the udp stack has data which means there
    // is pending datagram
    //
    while(udp_socket.hasPendingDatagrams())  
    {
        // Recieve the pending network datagram formatted in ANSI C's structure
        QNetworkDatagram datagram = udp_socket.receiveDatagram();
        qInfo() << "Read: " << datagram.data() << " from " << datagram.senderAddress() << datagram.senderPort();
    }  
}

void scl::CUdpWorker::broadcast()
{
    qInfo() << "Broadcating ...";
    udp_timer.start();
}
