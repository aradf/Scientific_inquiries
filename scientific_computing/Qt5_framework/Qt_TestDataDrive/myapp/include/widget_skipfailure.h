#ifndef WIDGET_SKIPFAILURE_H
#define WIDGET_SKIPFAILURE_H

#include <QObject>
#include <QDebug>
#include <QTest>

class CWidgetSkipFailure : public QObject
{
    Q_OBJECT
public:
    explicit CWidgetSkipFailure(QObject *parent = nullptr);

signals:

public slots:

private slots:
    void test();
    void test_data();
    void testFail();

};

#endif // WIDGET_H
