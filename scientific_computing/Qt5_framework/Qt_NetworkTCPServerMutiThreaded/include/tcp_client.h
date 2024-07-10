#ifndef CLIENT_H
#define CLIENT_H

#include <QObject>
#include <QDebug>
#include <QRunnable>
#include <QThread>
#include <QTcpSocket>

class CClient : public QObject, public QRunnable
{
    Q_OBJECT
public:
    explicit CClient(QObject *parent = nullptr, qintptr some_handle = 0);

signals:

public slots:

    // QRunnable interface
public:
    void run();
private:
    qintptr handle;
};

#endif // CLIENT_H
