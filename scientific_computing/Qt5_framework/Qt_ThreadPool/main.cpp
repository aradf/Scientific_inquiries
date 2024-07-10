#include <QCoreApplication>
#include <QDebug>
#include <QThread>
#include <QThreadPool>
#include <QDateTime>


#include <../include/first.h>
#include <../include/thread_counter.h>

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
        QThread::currentThread()->setObjectName("main");
        QThreadPool * ptr_threadPool = QThreadPool::globalInstance();
        qInfo() << ptr_threadPool->maxThreadCount() << " Threads";

        for (int i = 0; i < 16; i++)
        {
            CThreadCounter * thread_counter = new CThreadCounter();
            thread_counter->setAutoDelete(true);
            ptr_threadPool->start(thread_counter);
        }


    return a.exec();
}
