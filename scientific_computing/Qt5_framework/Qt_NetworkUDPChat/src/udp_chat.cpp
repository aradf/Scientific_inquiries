#include "udp_chat.h"

CChat::CChat(QObject *parent) : QObject(parent)
{
    // bind to a port == udp_port; 
    // if(!udp_socket.bind(udp_port, QUdpSocket::ReuseAddressHint)) //ShareAddress has some issues on windows
    if(!udp_socket.bind(udp_port, QUdpSocket::ShareAddress)) //ShareAddress has some issues on windows
    {
        qInfo() << udp_socket.errorString();
    }
    else
    {
        qInfo() << "Started on: " << udp_socket.localAddress() << ":" << udp_socket.localPort();
        // connect signals from QUdpSocket (udp_socket) to consuming slot in function CChat::readReady
        connect(&udp_socket,&QUdpSocket::readyRead,this, &CChat::readyRead);
    }
}

void CChat::command(QString value)
{
    QString message = name + ": ";

    if(name.isEmpty())
    {
        name = value;
        message = name + ": joined";
        send(message);
        return;
    }

    message.append(value);
    send(message);
}

void CChat::send(QString value)
{
    // convert the value to ByteArray
    QByteArray data = value.toLatin1();

    // set up a structure for the data gram with 'data', 'broadcast', and 'port'
    QNetworkDatagram datagram(data, QHostAddress::Broadcast, udp_port);

    // write the datagram to the soket.
    if(!udp_socket.writeDatagram(datagram))
    {
        qInfo() << udp_socket.errorString();
    }
}

void CChat::readyRead()
{
    while (udp_socket.hasPendingDatagrams())
    {
        QNetworkDatagram datagram = udp_socket.receiveDatagram();
        qInfo() << datagram.data();
    }
}

