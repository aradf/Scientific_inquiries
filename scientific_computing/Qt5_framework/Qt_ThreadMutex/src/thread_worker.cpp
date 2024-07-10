#include "thread_worker.h"


CThreadWorker::CThreadWorker(QObject *parent, 
                             CThreadCounter * some_counter,
                             QMutex * some_mutex) : QObject(parent), QRunnable()
                                                                

{
    this->thread_counter = some_counter;
    this->thread_mutex   = some_mutex;
}

void CThreadWorker::run()
{
    if(!thread_counter)
        return;

    qInfo() << this << " starting ...";

    for (int i = 0; i < 100; i++)
    {
        //Without our mutex, the count is wildy out of control
        //Only lock for short term durations!

        QMutexLocker locker(thread_mutex);
        thread_counter->increment();
        qInfo() << this << " Count: " << thread_counter->return_count();
        thread_counter->decrement();        
    }

    qInfo() << this << " Finished ...";

}