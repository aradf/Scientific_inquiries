#include "tcp_client.h"

CClient::CClient(QObject *parent) : QObject(parent)
{
    // connect signal QTcpSocket::connected (tcp_socket) to consumer QTcpSocket::connected
    connect(&tcp_socket, &QTcpSocket::connected, this, &CClient::connected);
    connect(&tcp_socket, &QTcpSocket::disconnected, this, &CClient::disconnected);
    connect(&tcp_socket, &QTcpSocket::stateChanged, this, &CClient::stateChanged);
    connect(&tcp_socket, &QTcpSocket::readyRead, this, &CClient::readyRead);
    
    // The QOverload is a Qt templated class with datatype 'QAbstractSocket::SocketError'
    connect(&tcp_socket, 
            QOverload<QAbstractSocket::SocketError>::of(&QAbstractSocket::error),
            this,
            &CClient::error); 

    // set up the QNetworkProxy structure with the values of HttpProxy,
    // ip address and port number;

    QNetworkProxy proxy(QNetworkProxy::HttpProxy,"134.209.67.109",26000);
    // QNetworkProxy proxy(QNetworkProxy::HttpProxy,"171.100.204.126",50858);
    // QNetworkProxy proxy(QNetworkProxy::HttpProxy,"185.44.26.217",43097);
    // set authenticators
    // proxy.setuser("username");
    // proxy.setPassword("password");

    // per application
    // QNetworkProxy::setApplicationproxy(proxy);

    // per socket;
    tcp_socket.setProxy(proxy);

}

void CClient::connectToHost(QString some_host, quint16 some_port)
{
    if(tcp_socket.isOpen()) 
        disconnect();

    qInfo() << "Connecting to: " << some_host << " on port " << some_port;

    tcp_socket.connectToHost(some_host, some_port);
}

// The disconnection could be many reasons/
void CClient::disconnect()
{
    tcp_socket.close();
}

void CClient::connected()
{
    qInfo() << "Connected!";

    qInfo() << "Sending";
    // tcp_socket.write("HELLO\r\n");
    QByteArray data;
    data.append("GET /get HTTP/1.1\r\n");
    data.append("User-Agent: Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)\r\n");
    data.append("Host: local\r\n");
    data.append("Connection: Close\r\n");
    data.append("\r\n");
    tcp_socket.write(data);

    tcp_socket.waitForBytesWritten();
}

void CClient::disconnected()
{
    qInfo() << "Disconnected";
}

void CClient::error(QAbstractSocket::SocketError socket_error)
{
    qInfo() << "Error:" << socket_error << " " << tcp_socket.errorString();
}

void CClient::stateChanged(QAbstractSocket::SocketState socket_state)
{
    QMetaEnum metaEnum = QMetaEnum::fromType<QAbstractSocket::SocketState>();
    qInfo() << "State: " << metaEnum.valueToKey(socket_state);
}

void CClient::readyRead()
{
    qInfo() << "Data from: " << sender() << " bytes: " << tcp_socket.bytesAvailable() ;
    qInfo() << "Data: " << tcp_socket.readAll();
}
