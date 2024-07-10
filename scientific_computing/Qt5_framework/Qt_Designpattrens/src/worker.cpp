#include "worker.h"

CWorker::CWorker(QObject *parent) : QObject(parent)
{
    m_busy_ = false;
}

bool CWorker::is_busy()
{
    return m_busy_;
}

void CWorker::timeout()
{
    m_busy_ = false;
    emit finished();
}

void CWorker::work(int value)
{
    m_busy_ = true;
    qInfo() << "Starting work: " << QString::number(value);
    int num = QRandomGenerator::global()->bounded(1000,5000);
    m_timer_.singleShot(num,this,&CWorker::timeout);

    emit started();
}
