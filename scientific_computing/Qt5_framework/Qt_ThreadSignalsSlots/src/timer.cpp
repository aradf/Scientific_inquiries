#include "timer.h"

CTimer::CTimer(QObject *parent) : QObject(parent)
{

}

void CTimer::set_interval(int value)
{
    interval = value;
}

void CTimer::started()
{
    qInfo() << "Timer started on " << QThread::currentThread();
    this->thread()->msleep(interval);
    emit timeout();
}

