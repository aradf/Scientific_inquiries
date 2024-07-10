#include "consumer.h"

CConsumer::CConsumer(QObject *parent) : QObject(parent)
{

}

void CConsumer::set_data(QList<int> *some_data)
{
    this->list_data = some_data;
}

void CConsumer::set_mutex(QMutex *mutex)
{
    this->mutex = mutex;
}

void CConsumer::set_condition(QWaitCondition * some_condition)
{
    this->condition = some_condition;
}

void CConsumer::start()
{
    qInfo() << "Starting consumer on: " << this->thread();

    do
    {
        qInfo() << "Consuming on: " << this->thread();

        mutex->lock();

        //TO DO - do something with the data
        list_data->clear();

        //pause
        condition->wait(mutex);

        mutex->unlock();

    } while(true);
}


