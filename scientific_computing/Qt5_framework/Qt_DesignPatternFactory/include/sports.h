#ifndef SPORTS_H
#define SPORTS_H

#include <QObject>
#include <QDebug>
#include "car.h"

class sports : public CCar
{
    Q_OBJECT
public:
    explicit sports(QObject *parent = nullptr);

signals:

public slots:

    // car interface
public:
    void drive();
};

#endif // SPORTS_H
