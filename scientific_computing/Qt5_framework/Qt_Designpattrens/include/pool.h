#ifndef POOL_H
#define POOL_H

#include <QObject>
#include <QDebug>
#include <QVector>
#include <QTimer>
#include "worker.h"

class CPool : public QObject
{
    Q_OBJECT
public:
    explicit CPool(QObject *parent = nullptr);
    ~CPool();

signals:

public slots:
    void work(int value);
    void started();
    void finished();
    void checkwork();

private:
    QVector<CWorker*> m_workers_;
    QVector<int> m_work_;
    QTimer m_timer_;
};

#endif // POOL_H
