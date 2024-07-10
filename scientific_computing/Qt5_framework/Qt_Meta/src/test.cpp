#include "../include/test.h"

scl::CTest::CTest(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CTest invoked ...";
    this->number_ = 0;
    timer_.setInterval(5000);

    // Connect this class timer_ data member of Qt's QTimer class with the QTimer's timeout signle; 
    // Connect the pointer this class to the CTest's timeout slot;
    // Every time the QTimer's timeout signal is emitted; The CTest::timeout slot will consume
    // the signal.
    QObject::connect(&timer_,&QTimer::timeout, this, &CTest::timeout);
}

scl::CTest::~CTest()
{
    qInfo() << this << "Destructor: CFirst invoked ...";
}

void scl::CTest::timeout()
{
    this->number_++;
    qInfo() << QTime::currentTime().toString(Qt::DateFormat::SystemLocaleLongDate);
    qInfo() << "Counting ..." << this->number_;
    if(number_ == 5)
    {
        timer_.stop();
        qInfo() << "Complete";
    }
}

void scl::CTest::do_stuff()
{
    this->timer_.start();
}