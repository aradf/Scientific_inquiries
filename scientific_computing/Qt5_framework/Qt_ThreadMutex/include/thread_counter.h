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

    void increment();
    void decrement();
    int return_count();

signals:

public slots:

private:
    int value = 0;    

};

#endif // COUNTER_H
