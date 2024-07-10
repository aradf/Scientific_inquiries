#ifndef TIMER_H
#define TIMER_H

#include <QObject>
#include <QDebug>
#include <QThread>

class CTimer : public QObject
{
    Q_OBJECT
public:
    explicit CTimer(QObject *parent = nullptr);
    void set_interval(int value);

signals:
    // emit a signal to be consumed by some slot.
    void timeout();

public slots:
    // consume a signal to be emitted by some signal.
    void started();

private:
    int interval = 1000;

};

#endif // TIMER_H
