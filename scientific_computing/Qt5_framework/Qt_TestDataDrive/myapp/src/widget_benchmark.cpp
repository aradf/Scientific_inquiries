#include "../include/widget_benchmark.h"

CWidgetBenchmark::CWidgetBenchmark(QObject *parent) : QObject(parent)
{

}

void CWidgetBenchmark::testFor()
{
    QVector<int> list;
    list.fill(0,100);

    //Called multiple times!
    QBENCHMARK
    {
        for (int i = 0; i < list.size();i++)
        {
            //Do Stuff
        }
    }
}

void CWidgetBenchmark::testForEach()
{
    QVector<int> list;
    list.fill(0,100);

    //Called multiple times!
    QBENCHMARK
    {
        foreach(int value, list)
        {
            //Do Stuff
        }
    }
}

void CWidgetBenchmark::testString()
{
    QString him = "Bryan";
    QString her = "Tammy";

    QBENCHMARK
    {
        int ret = him.compare(her);
    }
}

void CWidgetBenchmark::testComp()
{
    QString him = "Bryan";
    QString her = "Tammy";

    QBENCHMARK
    {
        QCOMPARE(him, her);
    }
}
