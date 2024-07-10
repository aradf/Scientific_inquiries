#ifndef WIDGET_H
#define WIDGET_H

#include <QObject>
#include "test.h"

class widget : public QObject
{
    Q_OBJECT
public:
    explicit widget(QObject *parent = nullptr);
    void make_changes(QString value);
signals:

public slots:
};

#endif // WIDGET_H
