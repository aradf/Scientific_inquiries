#ifndef WORKER_H
#define WORKER_H

#include <QObject>
#include <QRunnable>
#include <QDebug>
#include <QRandomGenerator>
#include <QMutex>
#include <QMutexLocker>

#include "thread_counter.h"

class CThreadWorker : public QObject, public QRunnable
{
    Q_OBJECT
public:
    explicit CThreadWorker(QObject * parent = nullptr, 
                           CThreadCounter * some_counter = nullptr,
                           QMutex * some_mutex = nullptr);
    void run();

signals:

public slots:

private:
    CThreadCounter * thread_counter;
    QMutex * thread_mutex;

};

#endif // MANAGER_H
