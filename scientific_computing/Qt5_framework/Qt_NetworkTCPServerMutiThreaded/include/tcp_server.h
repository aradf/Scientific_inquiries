#ifndef SERVER_H
#define SERVER_H

#include <QObject>
#include <QDebug>
#include <QTcpServer>
#include <QTcpSocket>
#include <QThreadPool>
#include <QThread>
#include "tcp_client.h"

class CServer : public QTcpServer
{
    Q_OBJECT
public:
    explicit CServer(QObject *parent = nullptr);

signals:

public slots:
    void start(quint16 port);
    void quit();


    // QTcpServer interface
protected:
    // Not version friendly!!!
    // Since this 'incomingConnection' is declared virutal.  The parent class and the child class both have
    // a function member with the same name.  The virtual function member in the child class is to be invoked. 
    virtual void incomingConnection(qintptr handle) Q_DECL_OVERRIDE;

private:
    QThreadPool thread_pool;
};

#endif // SERVER_H
