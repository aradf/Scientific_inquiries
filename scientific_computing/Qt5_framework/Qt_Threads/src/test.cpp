#include "test.h"

CTest::CTest(QObject *parent) : QObject(parent)
{
    qInfo() << this << "CTest Constructor ";
}

CTest::~CTest()
{
    qInfo() << this << "CTest destructer: ";
}

void CTest::timeout()
{
    // Consumes or recieves a trigger ();
    qInfo() << QDateTime::currentDateTime().toString() << " on " << QThread::currentThread();
    
}

void CTest::start()
{
    // consume or recieves a trigger time out from Qtimer and is connected to CTest::timeout
    QObject::connect(&timer, &QTimer::timeout, this, &CTest::timeout);

    timer.setInterval(1000);
    timer.start();
}
