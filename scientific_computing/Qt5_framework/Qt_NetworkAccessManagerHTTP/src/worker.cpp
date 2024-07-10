#include "worker.h"

CWorker::CWorker(QObject *parent) : QObject(parent)
{
    // Connect signal QNetworkAccessManager::authenticationRequired (network_accessManager) to 
    // CWorker::authenticationRequired consuming slot.
    connect(&network_accessManager, &QNetworkAccessManager::authenticationRequired, this, &CWorker::authenticationRequired);
    connect(&network_accessManager, &QNetworkAccessManager::encrypted, this, &CWorker::encrypted);
    connect(&network_accessManager, &QNetworkAccessManager::networkAccessibleChanged, this, &CWorker::networkAccessibleChanged);
    connect(&network_accessManager, &QNetworkAccessManager::preSharedKeyAuthenticationRequired, this, &CWorker::preSharedKeyAuthenticationRequired);
    connect(&network_accessManager, &QNetworkAccessManager::proxyAuthenticationRequired, this, &CWorker::proxyAuthenticationRequired);
    connect(&network_accessManager, &QNetworkAccessManager::sslErrors, this, &CWorker::sslErrors);
}

void CWorker::get(QString location)
{
    qInfo() << "Getting from server...";
    // replay is an object variable == 0x1234; pointing 'QNetowrkReplay'
    QNetworkReply* reply = network_accessManager.get( QNetworkRequest( QUrl ( location ) ) );
    connect(reply, &QNetworkReply::readyRead, this, &CWorker::readyRead);
}

void CWorker::post(QString location, QByteArray data)
{
    qInfo() << "Posting to server...";
    
    QNetworkRequest request = QNetworkRequest( QUrl ( location ) );
    request.setHeader( QNetworkRequest::ContentTypeHeader, "text/plain");

    QNetworkReply* reply = network_accessManager.post( request, data );
    connect(reply, &QNetworkReply::readyRead, this, &CWorker::readyRead );
}

void CWorker::readyRead()
{
    qInfo() << "ReadyRead";

    // object_cast is an operator that performs data conversion to QNetworkReplay pointer object.
    // output pointer object of QNetworkReply class type; sender is pointer to a QObject;
    QNetworkReply* reply = qobject_cast<QNetworkReply*>( sender() );
    if(reply) 
        qInfo() << reply->readAll();
}

void CWorker::authenticationRequired(QNetworkReply *reply, QAuthenticator *authenticator)
{
    Q_UNUSED(reply);
    Q_UNUSED(authenticator);
    qInfo() << "authenticationRequired";
}

void CWorker::encrypted(QNetworkReply *reply)
{
    Q_UNUSED(reply);
    qInfo() << "encrypted";
}

void CWorker::finished(QNetworkReply *reply)
{
    Q_UNUSED(reply);
    qInfo() << "finished";
}

void CWorker::networkAccessibleChanged(QNetworkAccessManager::NetworkAccessibility accessible)
{
    Q_UNUSED(accessible);
    qInfo() << "networkAccessibleChanged";
}

void CWorker::preSharedKeyAuthenticationRequired(QNetworkReply *reply, QSslPreSharedKeyAuthenticator *authenticator)
{
    Q_UNUSED(reply);
    Q_UNUSED(authenticator);
    qInfo() << "preSharedKeyAuthenticationRequired";
}

void CWorker::proxyAuthenticationRequired(const QNetworkProxy &proxy, QAuthenticator *authenticator)
{
    Q_UNUSED(proxy);
    Q_UNUSED(authenticator);
    qInfo() << "proxyAuthenticationRequired";
}

void CWorker::sslErrors(QNetworkReply *reply, const QList<QSslError> &errors)
{
    Q_UNUSED(reply);
    Q_UNUSED(errors);
    qInfo() << "sslErrors";
}

