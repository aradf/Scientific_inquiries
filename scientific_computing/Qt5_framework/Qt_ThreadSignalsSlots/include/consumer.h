#ifndef CONSUMER_H
#define CONSUMER_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include "timer.h"

class CConsumer : public QObject
{
    Q_OBJECT
public:
    explicit CConsumer(QObject *parent = nullptr);

signals:

public slots:
    // slot function members to consume emitted signals.
    void timeout();
    void start();
    void finished();


private:
    QThread consumer_thread;
    CTimer consumer_timer;
};

#endif // CONSUMER_H
