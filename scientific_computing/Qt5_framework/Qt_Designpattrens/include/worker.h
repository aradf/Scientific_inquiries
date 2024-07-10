#ifndef WORKER_H
#define WORKER_H

#include <QObject>
#include <QTimer>
#include <QRandomGenerator>
#include <QDebug>

class CWorker : public QObject
{
    Q_OBJECT
public:
    explicit CWorker(QObject *parent = nullptr);
    bool is_busy();

signals:
    void started();
    void finished();


public slots:
    void timeout();
    void work(int value);

private:
    QTimer m_timer_;
    bool m_busy_;
};

#endif // WORKER_H
