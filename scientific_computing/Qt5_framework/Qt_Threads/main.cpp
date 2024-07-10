#include <QCoreApplication>
#include <QDebug>
#include <QThread>
#include <QDateTime>
#include <QTimer>
#include <QSharedPointer>

#include <../include/first.h>
#include <../include/test.h>

// 'static' means the shared_ptrThread object variable is global in 
// this file only.  It is a Qt's templated classs 'QWharedPointer' holding
// a QThread data type.
static QSharedPointer<QThread> shared_ptrThread;

void timeout()
{
    if (!shared_ptrThread.isNull())
    {
        qInfo() << "Time Out - stopping other thread from: " << QThread::currentThread();
        // .data() returns a raw pointer;
        shared_ptrThread.data()->quit();
    }
}

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

    qInfo() << "Current Thread (main): " << QThread::currentThread();
    qInfo() << "Application Thread: " << a.thread();

    // CTest my_test(&a);            // will fail at compilation time
    CTest my_test;
    qInfo() << "Timer Thread: (main)" << my_test.thread();

    QThread working_thread;
    shared_ptrThread.reset(&working_thread);
    // Move the my_test variable to a working thread;
    my_test.moveToThread(&working_thread);
    qInfo() << "Timer Thread (working): " << my_test.thread();

    my_test.start();
    qInfo() << "Working Thread State: " << working_thread.isRunning();

    working_thread.start();
    qInfo() << "Working Thread State: " << working_thread.isRunning();

    QTimer timer;
    timer.singleShot(5000, &timeout);

    qInfo() << "got it...";       



    return a.exec();
}
