#ifndef SEMAPHORE_WORKER_H
#define SEMAPHORE_WORKER_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <QRunnable>
#include <QSemaphore>
#include <QStringList>

class CSemahporeWorker : public QObject, public QRunnable
{
    Q_OBJECT
public:
    explicit CSemahporeWorker(QObject * parent = nullptr,
                              QStringList * some_dataList = nullptr,
                              QSemaphore * some_threadSemaphore = nullptr,
                              int some_position = -1);
    ~CSemahporeWorker();
    void run();

signals:

public slots:

private:
    QStringList * data_list;
    QSemaphore * thread_semaphore;
    int position;

};

#endif // MANAGER_H
