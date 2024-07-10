#ifndef SERVER_H
#define SERVER_H

#include <QObject>
#include <QDebug>
#include <QAbstractSocket>        // Interface/parent class provding base functionality fro all socket types.
#include <QTcpServer>             // Accepting incoming TCP connection
#include <QTcpSocket>             // Stablish tcp connection and transfer bytearrays of data.

// CServer is a public child of QObject
class CServer : public QObject
{
    // This macro allows for slots and signals
    Q_OBJECT
public:

    // explicit data conversion
    explicit CServer(QObject *parent = nullptr);

signals:

public slots:
    // consuming slots recieve a emitted signal.
    // Start the server
    void start();
    // tell the server to
    void quit();
    // establish new connection with clients
    void newConnection();
    // when things are disconnect
    void disconnected();
    // a connecting socket is providing data.
    void readyRead();

private:
    // object variable of QTcpServer
    QTcpServer tcp_server;

};

#endif // SERVER_H
