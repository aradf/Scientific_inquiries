#ifndef WIDGET_DATA_H
#define WIDGET_DATA_H

#include <QObject>
#include <QDebug>
#include <QTest>

class CWidgetData : public QObject
{
    Q_OBJECT
public:
    explicit CWidgetData(QObject *parent = nullptr);

signals:

public slots:

private slots:
    void testage_data();
    void testage();
};

#endif // WIDGET_H
