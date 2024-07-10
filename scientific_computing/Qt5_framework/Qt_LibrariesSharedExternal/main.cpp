#include <QCoreApplication>
#include <QDebug>

/*
 *  ldd ./runqtIODevice
 */

#include "Headers/mylib_global.h"
#include "Headers/mylib.h"

#include "include/first.h"

/**
 * https://doc.qt.io/qt-5/cmake-get-started.html
 * https://cmake.org/cmake/help/latest/guide/tutorial/Adding%20a%20Library.html
 * https://www.youtube.com/watch?v=l9CcfSRKeTM
 * https://www.youtube.com/watch?v=_5wbp_bD5HA
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
        mylib.test_library();
        QString name_library = mylib.get_name();

        scl::CFirst my_first;
        my_first.set_someString(name_library);
    }

    //some change

    return a.exec();
}
