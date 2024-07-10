#ifndef MANAGER_H
#define MANAGER_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include "thread_counter.h"

class CThreadManager : public QObject
{
    Q_OBJECT
public:
    explicit CThreadManager(QObject *parent = nullptr);
    void start();

signals:

public slots:
    void started();
    void finished();

private:
    QList<QThread*> threads_list;
};

#endif // MANAGER_H
