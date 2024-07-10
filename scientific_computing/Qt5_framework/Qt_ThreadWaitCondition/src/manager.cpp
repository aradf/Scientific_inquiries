#include "manager.h"

CManager::CManager(QObject *parent) : QObject(parent)
{
    // Signal is Qt::QueuedConnection emitted by the QThread's object variable producer_thread; consumed by the CProducer's producer object
    connect(&producer_thread,&QThread::started,&producer,&CProducer::start, Qt::QueuedConnection);
    // signal Qt::QueuedConnection emitted by CProducer's producer; consumed by CManager's this object.
    connect(&producer,&CProducer::ready,this,&CManager::ready, Qt::QueuedConnection);
    // signal is Qt::QueueConnection emitted by QThread's consumber_thread object variable; consumed by CConsumer's consumer object variable.
    connect(&consumer_thread,&QThread::started,&consumer,&CConsumer::start, Qt::QueuedConnection);

    // Use the QObject's setObjectName method to set the inherited name.
    producer_thread.setObjectName("Producer Thread");
    consumer_thread.setObjectName("Consumer Thread");
    this->thread()->setObjectName("Main Thread");

    producer.moveToThread(&producer_thread);
    consumer.moveToThread(&consumer_thread);

}

void CManager::start()
{
    producer.set_mutex(&mutex);
    producer.set_data(&list_data);

    consumer.set_mutex(&mutex);
    consumer.set_data(&list_data);
    consumer.set_condition(&condition);

    producer_thread.start();
    consumer_thread.start();
}

void CManager::ready()
{
    qInfo() << "Data is ready " << this->thread();
    condition.wakeAll();
}

