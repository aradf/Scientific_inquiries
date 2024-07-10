#ifndef WIDGET_BENCHMARK_H
#define WIDGET_BENCHMARK_H

#include <QObject>
#include <QDebug>
#include <QTest>

// Bench marking Code
// https://doc.qt.io/qt-5/qtest-overview.html#benchmarking-options

class CWidgetBenchmark : public QObject
{
    Q_OBJECT
public:
    explicit CWidgetBenchmark(QObject *parent = nullptr);

signals:

public slots:

private slots:
    void testFor();
    void testForEach();
    void testString();
    void testComp();

};

#endif // WIDGET_H
