#include "producer.h"

CProducer::CProducer(QObject *parent) : QObject(parent)
{

}

void CProducer::set_data(QList<int> * some_data)
{
    this->list_data = some_data;
}

void CProducer::set_mutex(QMutex * some_mutex)
{
    this->mutex = some_mutex;
}

void CProducer::start()
{
    do
    {
        qInfo() << "Producing on " << this->thread();

        int value = QRandomGenerator::global()->bounded(1000);
        mutex->lock();
        // list of data is inside lock block;
        list_data->append(value);

        // once length of buffer is 100 emit signal
        if(list_data->length() >= 100) 
            emit ready();

        mutex->unlock();
    } while(true);
}
