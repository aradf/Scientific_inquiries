#ifndef CHAT_H
#define CHAT_H

#include <QObject>
#include <QDebug>
#include <QUdpSocket>
#include <QNetworkDatagram>

// CChat is a public child of QObject
class CChat : public QObject
{
    // Gives access to signals and slots
    Q_OBJECT
public:
    // explicit data conversion; pointer object parent == 0x1234 (l-value)
    // *parent is (r-value) or content; use the '->' operator to invoke methods;
    explicit CChat(QObject *parent = nullptr);

signals:

public slots:
    // consuming functions connected to some emitted signals
    void command(QString value);
    void send(QString value);
    void readyRead();

private:
    QString name;
    QUdpSocket udp_socket;
    quint16 udp_port = 3971;
};

#endif // CHAT_H
