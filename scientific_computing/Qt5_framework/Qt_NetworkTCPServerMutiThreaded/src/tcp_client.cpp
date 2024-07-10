#include "tcp_client.h"

CClient::CClient(QObject *parent, qintptr some_handle) : QObject(parent), QRunnable ()
{
    this->handle = some_handle;
}

void CClient::run()
{

    qInfo() << this << " run " << QThread::currentThread();

    QTcpSocket* temp_socket = new QTcpSocket(nullptr);
    if(!temp_socket->setSocketDescriptor(handle))
    {
        qCritical() << temp_socket->errorString();
        delete temp_socket;
        return;
    }

    temp_socket->waitForReadyRead();
    QByteArray request = temp_socket->readAll();
    qInfo() << "Request : " << request;
    qInfo() << "Request Length: " << request.length();

    QByteArray data("Hello World!");
    QByteArray response;
    response.append("HTTP/1.1 200 OK\r\n");
    response.append("Content-Type: text/plain\r\n");
    response.append("Content-Length: " + QString::number(data.length()) + "\r\n");
    response.append("Connection: close\r\n");
    response.append("\r\n");
    response.append(data);
    qInfo() << "Outgoing data: " << data;

    temp_socket->write(response);
    temp_socket->waitForBytesWritten();
    temp_socket->close();
    temp_socket->deleteLater();

    qInfo() << this << " done " << QThread::currentThread();
}
