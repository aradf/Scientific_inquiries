#include <QCoreApplication>
#include <QDebug>

#include <../include/first.h>
#include <../include/worker.h>

/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);


    CWorker worker;

    worker.get("https://postman-echo.com/get?foo1=bar1&foo2=bar2");


    QByteArray data;
    data.append("param1=hello");
    data.append("&");
    data.append("param2=foo");

    worker.post("https://postman-echo.com/post",data);


    return a.exec();
}
