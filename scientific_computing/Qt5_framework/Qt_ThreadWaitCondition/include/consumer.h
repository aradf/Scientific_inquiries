#ifndef CONSUMER_H
#define CONSUMER_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>

class CConsumer : public QObject
{
    Q_OBJECT
public:
    explicit CConsumer(QObject *parent = nullptr);

    void set_data(QList<int>* some_data);
    void set_mutex(QMutex* some_mutex);
    void set_condition(QWaitCondition * some_condition);

signals:

public slots:
    void start();

private:
    QList< int > * list_data;
    QMutex * mutex;
    QWaitCondition * condition;


};

#endif // CONSUMER_H
