#include <QCoreApplication>
#include <QDebug>
#include "test.h"
#include "widget.h"

//Singleton pattern
//https://wiki.qt.io/Qt_thread-safe_singleton
//Watch for bug in singleton.h line 31

/*
    Need one and only one instance of a class
 */

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    test::instance()->name = "Bryan";

    qInfo() << test::instance()->name;

    for(int i = 0; i < 5; i++) {
        widget w;
        w.makeChanges("Widget: " + QString::number(i));
    }


    qInfo() << test::instance()->name;

    return a.exec();
}
