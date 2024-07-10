#include "pool.h"

CPool::CPool(QObject *parent) : QObject(parent)
{
    for(int i = 0; i < 5; i++) 
    {
        CWorker *w = new CWorker(this);
        w->setObjectName("CWorker: " + QString::number(i));
        connect(w,&CWorker::started,this, &CPool::started);
        connect(w,&CWorker::finished,this, &CPool::finished);

        m_workers_.append(w);
        qInfo() << "Worker ready: " << w->objectName();
    }

    connect(&m_timer_,&QTimer::timeout,this, &CPool::checkwork);
    m_timer_.setInterval(200);
    m_timer_.start();
}

CPool::~CPool()
{
    m_timer_.stop();
    qDeleteAll(m_workers_);
    m_workers_.clear();
}

void CPool::work(int value)
{
    m_work_.append(value);
    checkwork();
}

void CPool::started()
{
    CWorker *w = qobject_cast<CWorker*>(sender());
    qInfo() << "Started: " << w->objectName();
}

void CPool::finished()
{
    CWorker *worker = qobject_cast<CWorker*>(sender());
    qInfo() << "Finished: " << worker->objectName();
}

void CPool::checkwork()
{
    if(m_work_.isEmpty()) return;
    foreach(CWorker *worker_item, m_workers_) 
    {
        if(!worker_item->is_busy()) 
        {
            worker_item->work(m_work_.takeFirst());
            if(m_work_.isEmpty()) return;
        }
    }
}
