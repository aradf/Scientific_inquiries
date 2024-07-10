#include <QDebug>
#include <QtCore/QCoreApplication>

int main(int argc, char *argv[])
{
   QCoreApplication a(argc, argv);
   qInfo() << "Hello World Using QT CMake ...";

   return a.exec();
}