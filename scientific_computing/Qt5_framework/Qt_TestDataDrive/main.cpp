#include <QCoreApplication>
#include <QTest>
// #include "myapp/include/widget_data.h"
// #include "myapp/include/widget_benchmark.h"
#include "myapp/include/widget_skipfailure.h"

//Update .Pro file:  QT += testlib

//Replaces the main
//  QTEST_MAIN(CWidgetData);
//  QTEST_MAIN(CWidgetBenchmark)
QTEST_MAIN(CWidgetSkipFailure)
