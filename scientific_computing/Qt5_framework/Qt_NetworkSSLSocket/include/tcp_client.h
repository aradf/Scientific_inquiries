#ifndef CLIENT_H
#define CLIENT_H

#include <QObject>
#include <QDebug>
#include <QTcpSocket>
#include <QAbstractSocket>
#include <QMetaEnum>
#include <QNetworkProxy>
#include <QSslSocket>
#include <QSslPreSharedKeyAuthenticator>


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

    //SSL
    void encrypted();
    void encryptedBytesWritten(qint64 written);
    void modeChanged(QSslSocket::SslMode mode);
    void peerVerifyError(const QSslError &error);
    void preSharedKeyAuthenticationRequired(QSslPreSharedKeyAuthenticator *authenticator);
    void sslErrors(const QList<QSslError> &errors);

private:
    // QTcpSocket tcp_socket;
    QSslSocket ssl_socket;

};

#endif // CLIENT_H
