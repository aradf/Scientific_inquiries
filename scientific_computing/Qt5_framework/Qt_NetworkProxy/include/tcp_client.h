#ifndef CLIENT_H
#define CLIENT_H

#include <QObject>
#include <QDebug>
#include <QTcpSocket>
#include <QAbstractSocket>
#include <QMetaEnum>
#include <QNetworkProxy>

class CClient : public QObject
{
    Q_OBJECT
public:
    explicit CClient(QObject *parent = nullptr);

signals:

public slots:
    // function member to consume signals emitted from somewhere;
    void connectToHost(QString some_host, quint16 some_port);
    void disconnect();

private slots:
    // signals emitted from here to some consumer somewhere.
    void connected();
    void disconnected();
    void error(QAbstractSocket::SocketError socket_error);
    void stateChanged(QAbstractSocket::SocketState socket_state);
    void readyRead();

private:
    QTcpSocket tcp_socket;

};

#endif // CLIENT_H
