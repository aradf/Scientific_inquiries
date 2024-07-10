#ifndef WORKER_H
#define WORKER_H

#include <QObject>
#include <QDebug>
#include <QNetworkAccessManager>   // Allows the application to send netowrk request and receive replies.  It does the heavy lifting.
#include <QNetworkReply>           // This is a sequential-access QIODevice, Once data is read from the object it is no longer kept (response back from the server).
#include <QNetworkRequest>         // Holds a request to be sent with QNetworkAcessManager (Request made to the server).
#include <QAuthenticator>          // Used for authentication required and proxy authentication required.
#include <QNetworkProxy>           // povides method configuring network layer proxy support (working with a proxy server).

//CWork class is a public child of QObject.
class CWorker : public QObject
{
    Q_OBJECT
public:
    // Constructor with explicit data conversion; pointer object == 0x1234;
    // point to some location in memory block holding QObject type;
    // *parent is r-value; arrow operator invokes methods;
    explicit CWorker(QObject *parent = nullptr);

signals:
    // No signals to be emitted for some consuming slot.

public slots:
    // 'get' is a slot consuming data from the server.
    void get(QString location);
    // 'post' is a slot that puts data onto the server.
    // QByteArray is an array of machine readable bytes ASCII values.
    // https://www.asciitable.com/
    void post(QString location, QByteArray data);

private slots:
    // signals from https://doc.qt.io/qt-6/qnetworkaccessmanager.html
    // readyRead: slot consumes triggered signal when there is data from the socket.
    void readyRead();
    // QNetworkReply means once data is read from the object, it is no longer available.
    // replay == 0x1234 is a pointer object defereance to location on memory block;
    // *replay is the r-value; &replay == 0xABCD; arrow operator == '->' invoked function members
    // authenticationRequired:  The remote server does not know the server.
    void authenticationRequired(QNetworkReply *reply, QAuthenticator *authenticator);
    // encrypted: could be using an ssl socket.
    void encrypted(QNetworkReply *reply);
    // finished: handle the handshake.
    void finished(QNetworkReply *reply);
    // networkAccessibleChange:  The network is accessable
    void networkAccessibleChanged(QNetworkAccessManager::NetworkAccessibility accessible);
    // QSslPreSharedKeyAuthenticator class provides authen. data for pre-shared keys (PSK)
    // preSharedKeyAuthenticationRequired: ssl handshake
    void preSharedKeyAuthenticationRequired(QNetworkReply *reply, QSslPreSharedKeyAuthenticator *authenticator);
    // const means the address/referance can not be changed; proxy object can is passed by referance.
    // proxyAuthenticationRequired: We are talking to the proxy.
    void proxyAuthenticationRequired(const QNetworkProxy &proxy, QAuthenticator *authenticator);
    void sslErrors(QNetworkReply *reply, const QList<QSslError> &errors);

private:
    QNetworkAccessManager network_accessManager;

};

#endif // WORKER_H
