#include <QCoreApplication>
#include <QDebug>

#include "mylib/include/mylib.h"
#include "myapp/include/first.h"

/*
 *  ldd ./runqtIODevice
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
        Mylib mylib;
        mylib.test();
        QString some_string = mylib.get_string();

        scl::CFirst my_first;
        my_first.set_someString(some_string);
    }

    return a.exec();
}
