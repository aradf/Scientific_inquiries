#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QDateTime>
#include <QIODevice>
#include <QTextStream>
#include "logger.h"
#include "test.h"

#include <../include/first.h>

/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QString path = QDir::currentPath() + QDir::separator() + "test.txt";
    qInfo() << "Path: " << path;

    {
        qInfo() << "File: " << logger::filename;
        logger::attach();

        qInfo() << "test!";

        logger::logging = false;
        qInfo() << "Don't log this!";
        logger::logging = true;

        test t;
        t.testing();
    }

    return a.exec();
}