#include "tcp_client.h"

CClient::CClient(QObject *parent) : QObject(parent)
{
    // connect signal QTcpSocket::connected (ssl_socket) to consumer QTcpSocket::connected
    connect(&ssl_socket, &QTcpSocket::connected, this, &CClient::connected);
    connect(&ssl_socket, &QTcpSocket::disconnected, this, &CClient::disconnected);
    connect(&ssl_socket, &QTcpSocket::stateChanged, this, &CClient::stateChanged);
    connect(&ssl_socket, &QTcpSocket::readyRead, this, &CClient::readyRead);
    
    // The QOverload is a Qt templated class with datatype 'QAbstractSocket::SocketError'
    connect(&ssl_socket, 
            QOverload<QAbstractSocket::SocketError>::of(&QAbstractSocket::error),
            this,
            &CClient::error); 


    //Ssl
    connect(&ssl_socket,&QSslSocket::encrypted,this,&CClient::encrypted);
    connect(&ssl_socket,&QSslSocket::encryptedBytesWritten,this,&CClient::encryptedBytesWritten);
    connect(&ssl_socket,&QSslSocket::modeChanged,this,&CClient::modeChanged);
    connect(&ssl_socket,&QSslSocket::peerVerifyError,this,&CClient::peerVerifyError);
    connect(&ssl_socket,&QSslSocket::preSharedKeyAuthenticationRequired,this,&CClient::preSharedKeyAuthenticationRequired);
    connect(&ssl_socket, QOverload<const QList<QSslError> &>::of(&QSslSocket::sslErrors),this,&CClient::sslErrors);


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
    ssl_socket.setProxy(proxy);

}

void CClient::connectToHost(QString some_host, quint16 some_port)
{
    if(ssl_socket.isOpen()) 
        disconnect();

    qInfo() << "Connecting to: " << some_host << " on port " << some_port;

    // ssl_socket.connectToHost(some_host, some_port);  // Normal TCP
    ssl_socket.ignoreSslErrors();
    ssl_socket.setProtocol(QSsl::AnyProtocol);
    ssl_socket.connectToHostEncrypted(some_host, some_port);
}

// The disconnection could be many reasons/
void CClient::disconnect()
{
    ssl_socket.close();
}

void CClient::connected()
{
    qInfo() << "Connected!";

    qInfo() << "Sending";
    // ssl_socket.write("HELLO\r\n");
    QByteArray data;
    data.append("GET /get HTTP/1.1\r\n");
    data.append("User-Agent: Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)\r\n");
    data.append("Host: local\r\n");
    data.append("Connection: Close\r\n");
    data.append("\r\n");
    ssl_socket.write(data);

    ssl_socket.waitForBytesWritten();
}

void CClient::disconnected()
{
    qInfo() << "Disconnected";
}

void CClient::error(QAbstractSocket::SocketError socket_error)
{
    qInfo() << "Error:" << socket_error << " " << ssl_socket.errorString();
}

void CClient::stateChanged(QAbstractSocket::SocketState socket_state)
{
    QMetaEnum metaEnum = QMetaEnum::fromType<QAbstractSocket::SocketState>();
    qInfo() << "State: " << metaEnum.valueToKey(socket_state);
}

void CClient::readyRead()
{
    qInfo() << "Data from: " << sender() << " bytes: " << ssl_socket.bytesAvailable() ;
    qInfo() << "Data: " << ssl_socket.readAll();
}

void CClient::encrypted()
{
    qInfo() << "Encrypted";
}

void CClient::encryptedBytesWritten(qint64 written)
{
    qInfo() << "encryptedBytesWritten: " << written;
}

void CClient::modeChanged(QSslSocket::SslMode mode)
{
    //Qt does not support this!!!
    //QMetaEnum metaEnum = QMetaEnum::fromType<QSslSocket::SslMode>();
    //qDebug() << "SSL Mode: " << metaEnum.valueToKey(mode);
    qInfo() << "SslMode: " << mode;
}

void CClient::peerVerifyError(const QSslError &error)
{
    qInfo() << "peerVerifyError: " << error.errorString();
}

void CClient::preSharedKeyAuthenticationRequired(QSslPreSharedKeyAuthenticator *authenticator)
{
    qInfo() << "Preshared key required!";
    // QSslPreSharedKeyAuthenticator lookup if needed
    // authenticator->
}

void CClient::sslErrors(const QList<QSslError> &errors)
{
    qInfo() << "SSL Errors!";
    foreach(QSslError e, errors)
    {
        qInfo() << e.errorString();
    }
}