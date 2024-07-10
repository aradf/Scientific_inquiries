#include <QCoreApplication>
#include <QDebug>
#include <QThread>
#include <QThreadPool>
#include <QMutex>
#include <QDateTime>
#include <QSemaphore>

#include <../include/first.h>
#include <../include/thread_counter.h>
#include <../include/thread_worker.h>
#include <../include/semaphore_worker.h>

/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    {
        qInfo() << "App Thread: " << a.thread();
        qInfo() << "Current Thread: " << QThread::currentThread();

        Q_ASSERT(a.thread() == QThread::currentThread());

        qInfo() << "Running: " << QThread::currentThread()->isRunning();
        qInfo() << "loopLevel: " << QThread::currentThread()->loopLevel();
        qInfo() << "stackSize: " << QThread::currentThread()->stackSize();
        qInfo() << "isFinished: " << QThread::currentThread()->isFinished();
        qInfo() << "Before: " << QDateTime::currentDateTime().toString();
        QThread::sleep(2);
        qInfo() << "After: " << QDateTime::currentDateTime().toString();

        qInfo() << "got it...";       
    }

    {
        CThreadCounter thread_counter;
        QMutex mutex(QMutex::Recursive);
        QThreadPool * ptr_threadPool = QThreadPool::globalInstance();

        qInfo() << "Count: " << thread_counter.return_count();

        for (int i = 0;i < ptr_threadPool->maxThreadCount();i++)
        {
            //having a parent causes issues on some compilers and platforms
            //Worker* worker = new Worker(&a, &counter,&mutex);
            CThreadWorker* thread_worker = new CThreadWorker(nullptr, &thread_counter,&mutex);
            thread_worker->setAutoDelete(true);
            ptr_threadPool->start(thread_worker);
        }

        ptr_threadPool->waitForDone();

        qInfo() << "Done, the count is: " << thread_counter.return_count();
    }

    {
        // semaphores
        QStringList data_list;
        for(int i = 0; i < 100; i++)
        {
            data_list.append(QString::number(i));
        }

        QThreadPool* thread_pool = QThreadPool::globalInstance();
        QSemaphore semaphores(100);

        for (int i = 0;i < data_list.length();i++)
        {
            //Worker* worker = new Worker(&a,&data, &sema, i); Causes an error in windows 10
            CSemahporeWorker * semaphore_worker = new CSemahporeWorker(nullptr,&data_list, &semaphores, i);
            semaphore_worker->setAutoDelete(true);
            thread_pool->start(semaphore_worker);
        }

        thread_pool->waitForDone();

        foreach(QString string_item, data_list)
        {
            qInfo() << string_item;
        }

    }


    return a.exec();
}
