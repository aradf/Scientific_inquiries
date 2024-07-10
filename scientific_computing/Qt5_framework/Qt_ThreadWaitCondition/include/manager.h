#ifndef MANAGER_H
#define MANAGER_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include "producer.h"
#include "consumer.h"

class CManager : public QObject
{
    Q_OBJECT
public:
    explicit CManager(QObject *parent = nullptr);

signals:

public slots:
    void start();
    void ready();

private:
    QList<int> list_data;
    QMutex mutex;
    QThread producer_thread;
    QThread consumer_thread;
    QWaitCondition condition;
    CProducer producer;
    CConsumer consumer;
};

#endif // MANAGER_H
