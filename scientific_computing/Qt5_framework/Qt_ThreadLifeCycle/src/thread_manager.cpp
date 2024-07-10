#include "thread_manager.h"

CThreadManager::CThreadManager(QObject *parent) : QObject(parent)
{
    for(int i = 0;i < 5;i++)
    {
        QThread* ptr_thread = new QThread(this);
        ptr_thread->setObjectName("Thread " + QString::number(i));
        qInfo() << "Created: " << ptr_thread->objectName();

        connect(ptr_thread,&QThread::started, this, &CThreadManager::started);
        connect(ptr_thread,&QThread::finished, this, &CThreadManager::finished);

        threads_list.append(ptr_thread);
    }
}

void CThreadManager::start()
{
    qInfo() << "Starting...";

    foreach(QThread* thread_item, threads_list)
    {
        qInfo() << "Starting: " << thread_item->objectName();
        CThreadCounter* thread_counter = new CThreadCounter(); //NO PARENT!!!!
        thread_counter->moveToThread(thread_item);
        //c->start(); //Single Thread!!!

        connect(thread_item,&QThread::started, thread_counter,&CThreadCounter::start);
        thread_item->start();
    }
}

void CThreadManager::started()
{
    QThread* ptr_thread = qobject_cast<QThread*>(sender());
    if(!ptr_thread) 
        return;
    qInfo() << "Started: " << ptr_thread->objectName();
}

void CThreadManager::finished()
{
    QThread* thread = qobject_cast<QThread*>(sender());
    if(!thread) return;
    qInfo() << "Finished: " << thread->objectName();
}
