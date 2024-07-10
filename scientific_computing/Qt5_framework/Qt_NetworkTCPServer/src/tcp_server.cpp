#include "tcp_server.h"

CServer::CServer(QObject *parent) : QObject(parent)
{
    // Connect signal QTCPServer::newConnection to Slot CServer::newConnection
    connect(&tcp_server, &QTcpServer::newConnection, this, &CServer::newConnection);
}

void CServer::start()
{
    // binds the socket and listen to it for incoming connection on any ip address and port 2020;
    // this is just like a unix file to be openned.
    tcp_server.listen(QHostAddress::Any, 2020);
}

void CServer::quit()
{
    tcp_server.close();
}

void CServer::newConnection()
{
    //Same thread only!!!
    QTcpSocket * ptr_socket = tcp_server.nextPendingConnection();
    connect(ptr_socket, &QTcpSocket::disconnected, this, &CServer::disconnected);
    connect(ptr_socket, &QTcpSocket::readyRead, this, &CServer::readyRead);

    qInfo() << "Connected" << ptr_socket;
}

void CServer::disconnected()
{
    // ptr_socket is pointer object == 0x1234 (l-value);  dereferance *pointer to a location
    // on memory block with r-value of QTCPSocket; arrow operator '->' to invoke function members;
    // qobject_cast performs data conversion operating on QObject pointer parameter being data converted QTcpSocket pointers;  
    QTcpSocket* ptr_socket = qobject_cast< QTcpSocket* >( sender() );
    qInfo() << "Disconnected" << ptr_socket;
    qInfo() << "Parent" << ptr_socket->parent();

    ptr_socket->deleteLater();
    //delete socket;
}

void CServer::readyRead()
{
    QTcpSocket* socket = qobject_cast<QTcpSocket*>(sender());
    qInfo() << "ReadyRead" << socket;
    qInfo() << socket->readAll();
}
