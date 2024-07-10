#include "thread_counter.h"

CThreadCounter::CThreadCounter()
{

}

CThreadCounter::~CThreadCounter()
{

}

void CThreadCounter::run()
{
    qInfo() << "Running: " << QThread::currentThread();

    for(int i = 0; i < 20; i++)
    {
        qInfo() << QThread::currentThread()->objectName() << " = " << i;
        // static_cast is an operator that forces one data type to another;
        // in this case the data type is 'unsigned long' and paramter's datatype
        // is double;
        auto value = static_cast<unsigned long>(QRandomGenerator::global()->bounded(500));
        QThread::currentThread()->msleep(value);
    }

    qInfo() << "Finsihed: " << QThread::currentThread();

}
