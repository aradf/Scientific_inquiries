#include "consumer.h"

CConsumer::CConsumer(QObject *parent) : QObject(parent)
{
    qInfo() << "Created on " << QThread::currentThread();

    consumer_thread.setObjectName("Consumer Timer Thread");
    //Qt::AutoConnection - runs on both
    //Qt::QueuedConnection - runs on main thread
    //Qt::DirectConnection - runs on thread
    //Qt::BlockingQueuedConnection - blocks
    //Qt::UniqueConnection - combined with others

    // connect the signal CTimer::timeout using consumer_timer to be consumed with CConsumer::timeout slot
    // connect the signal QThread::started usig consumer_thread to be consumed by the CTimer::started slot.
    QObject::connect(&consumer_timer, &CTimer::timeout,this, &CConsumer::timeout,Qt::QueuedConnection);
    QObject::connect(&consumer_thread, &QThread::started,&consumer_timer, &CTimer::started,Qt::QueuedConnection);
    QObject::connect(&consumer_thread, &QThread::finished,this, &CConsumer::finished,Qt::QueuedConnection);
}

void CConsumer::timeout()
{
    qInfo() << "CConsumer timeout on " << QThread::currentThread();
    consumer_thread.quit();
}

void CConsumer::start()
{
    qInfo() << "CConsumer started on " << QThread::currentThread();
    consumer_timer.set_interval(500);
    consumer_timer.moveToThread(&consumer_thread);
    consumer_thread.start();
}

void CConsumer::finished()
{
    qInfo() << "Thread finished on " << QThread::currentThread();
}

