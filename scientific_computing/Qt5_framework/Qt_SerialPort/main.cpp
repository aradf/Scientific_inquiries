#include <QApplication>
#include <QtCore>
// #include <QtCore/QCoreApplication>   // Gives access to QCoreApplication.
#include <QDebug>                    // Gives access to qDebug, qInfo.
#include <QProcessEnvironment>       // Gives access to QProcessEnvironment
#include <QSerialPort>
#include <QSerialPortInfo>
#include <QTime>
#include <QTimer>


#include "include/serial_port.h"                  // gives access to serial port
#include "include/my_dialog.h"                    // Gives access to MyDialog

/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */
int main (int argc, char * argv[])
{
    qInfo() << "INFO: Hello World ...";

    QApplication app(argc, argv);
    MyDialog my_dialog;
    my_dialog.setMaximumSize(QWIDGETSIZE_MAX/2, QWIDGETSIZE_MAX/2);
    my_dialog.setMinimumSize(0, 0);
    my_dialog.show();
    return my_dialog.exec();
}