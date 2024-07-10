#ifndef MYTHREAD_H
#define MYTHREAD_H

#include <QThread>
#include <QTcpSocket>
#include <QDebug>

class MyThread : public QThread
{
    Q_OBJECT
public:
    explicit MyThread(int ID, QObject *parent = 0);
    void run();
    
signals:
    void error(QTcpSocket::SocketError socketerror);
    
public slots:
    void readyRead();
    void disconnected();

public slots:
    QTcpSocket *socket;
    int socket_descriptor;


};



#endif