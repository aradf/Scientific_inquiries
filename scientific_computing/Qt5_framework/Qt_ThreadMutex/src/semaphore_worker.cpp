#include "semaphore_worker.h"

CSemahporeWorker::CSemahporeWorker(QObject *parent,
                                   QStringList * some_dataList,
                                   QSemaphore * some_threadSemaphore,
                                   int some_position) : QObject(parent)
{
    this->data_list = some_dataList;
    this->thread_semaphore = some_threadSemaphore;
    this->position = some_position;

}

CSemahporeWorker::~CSemahporeWorker()
{

}

void CSemahporeWorker::run()
{
    if(!data_list || !thread_semaphore)
    {
        qInfo() << "Missing pointers!";
        return;
    }

    QString temp_string;
    temp_string.sprintf("%08p", QThread::currentThread());
    thread_semaphore->acquire(1);
    data_list->replace(position,QString::number(position) + " - " + temp_string);
    thread_semaphore->release();

    qInfo() << temp_string << " Finished " << position;
}