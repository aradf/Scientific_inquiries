#ifndef PRODUCER_H
#define PRODUCER_H

#include <QObject>
#include <QMutex>
#include <QRandomGenerator>
#include <QThread>
#include <QDebug>

//CProducer is a public child of QObject class
class CProducer : public QObject
{
    // Added for access t slots and signals
    Q_OBJECT
public:
    // Explciit is for data conversion. Constructor 'CProducer' has a parameter
    // pointer object like 'parent' == ox1234; It points to some location on
    // memory block holding a CProducer class type. *parent is the r-value
    // '->' or arrow operator invokes methods; &parent == 0xABCD
    explicit CProducer(QObject *parent = nullptr);

    // method set_data returns a void; It's parameters are object pointer 'data' 
    // (l-value);  it is Qt's templated QList holding integers;
    void set_data(QList<int> * some_data);

    // set_mutex returns a void;  It's parameter is pointer object 'mutex' == 0x1234
    // l-value; of Qt's class type QMutex; *mutex is content (r-value) temp data;
    void set_mutex(QMutex * some_mutex);

signals:
    // emitted signal to be consumed by some slot later.
    void ready();

public slots:
    // consuming public slot to recieve emitted singal from some where.
    void start();

private:
    // pointer object of Qt's templted QList holding integers;
    QList<int> * list_data;
    QMutex * mutex;


};

#endif // PRODUCER_H
