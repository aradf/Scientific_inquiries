#ifndef THREAD_COUNTER_H
#define THREAD_COUNTER_H

#include <QObject>
#include <QDebug>
#include <QRunnable>
#include <QThread>
#include <QRandomGenerator>

class CThreadCounter : public QRunnable
{
   
public:
    explicit CThreadCounter();
    ~CThreadCounter();

    void run();
};

#endif // COUNTER_H
