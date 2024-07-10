#include <QCoreApplication>
#include <QDebug>
#include <QRandomGenerator>

#include <../include/first.h>
#include "test.h"
#include "widget.h"

//Singleton pattern
//https://wiki.qt.io/Qt_thread-safe_singleton
//Watch for bug in singleton.h line 31

/*
    Need one and only one instance of a class
 */

/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    {
        // Single Pattern;
        quint32 value = QRandomGenerator::global()->generate();
    }

    {
        CTest::instance()->name_ = "Yonder";

        qInfo() << CTest::instance()->name_;

        for(int i = 0; i < 5; i++) 
        {
            widget my_widget;
            my_widget.make_changes("Widget: " + QString::number(i));
        }

        qInfo() << CTest::instance()->name_;
    }

    return a.exec();
}
