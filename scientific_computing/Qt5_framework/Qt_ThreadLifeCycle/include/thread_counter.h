#ifndef THREAD_COUNTER_H
#define THREAD_COUNTER_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <QRandomGenerator>

class CThreadCounter : public QObject
{
    Q_OBJECT
public:
    explicit CThreadCounter(QObject *parent = nullptr);

signals:

public slots:
    void start();
};

#endif // COUNTER_H
