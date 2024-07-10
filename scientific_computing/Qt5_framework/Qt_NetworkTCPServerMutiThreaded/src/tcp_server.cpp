#include "tcp_server.h"

CServer::CServer(QObject *parent) : QTcpServer(parent)
{
    thread_pool.setMaxThreadCount(20);
    qInfo() << "Threads: " << thread_pool.maxThreadCount();
}

void CServer::start(quint16 port)
{
    qInfo() << this << " start " << QThread::currentThread();

    if(this->listen(QHostAddress::Any, port))
    {
        qInfo() << "CServer started on " << port;
    }
    else
    {
        qCritical() << this->errorString();
    }
}

void CServer::quit()
{
    this->close();
    qInfo() << "CServer Stopped!";
}

void CServer::incomingConnection(qintptr handle)
{
    // Not Version friendly!!!
    // Please note this is a virtual function member and declared virtual, so this function
    // member is intended to be invoked.
    qInfo() << "Incomming Connection " << handle << " on " << QThread::currentThread();
    CClient* temp_client = new CClient(nullptr, handle);
    temp_client->setAutoDelete(true);
    thread_pool.start(temp_client);
}
