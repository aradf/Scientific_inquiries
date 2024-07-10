#include <QCoreApplication>
#include <QDebug>

/*
 *  ldd ./runqtIODevice

    linux-vdso.so.1 (0x00007fff65eeb000)
    libmylib.so => /home/montecarlo/Desktop/Qt5_framework/Qt_Libraries/build/mylib/libmylib.so (0x00007f34b76ab000)
    libQt5Core.so.5 => /usr/lib/x86_64-linux-gnu/libQt5Core.so.5 (0x00007f34b7162000)
    libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f34b6f80000)
    libgcc_s.so.1 => /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f34b6f65000)
    libc.so.6 => /usr/lib/x86_64-linux-gnu/libc.so.6 (0x00007f34b6d73000)
    libpthread.so.0 => /usr/lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f34b6d4e000)
    libz.so.1 => /usr/lib/x86_64-linux-gnu/libz.so.1 (0x00007f34b6d30000)
    libicui18n.so.66 => /usr/lib/x86_64-linux-gnu/libicui18n.so.66 (0x00007f34b6a31000)
    libicuuc.so.66 => /usr/lib/x86_64-linux-gnu/libicuuc.so.66 (0x00007f34b684b000)
    libdl.so.2 => /usr/lib/x86_64-linux-gnu/libdl.so.2 (0x00007f34b6845000)
    libpcre2-16.so.0 => /usr/lib/x86_64-linux-gnu/libpcre2-16.so.0 (0x00007f34b67c1000)
    libdouble-conversion.so.3 => /usr/lib/x86_64-linux-gnu/libdouble-conversion.so.3 (0x00007f34b67a9000)
    libglib-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libglib-2.0.so.0 (0x00007f34b667f000)
    libm.so.6 => /usr/lib/x86_64-linux-gnu/libm.so.6 (0x00007f34b6530000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f34b76b7000)
    libicudata.so.66 => /usr/lib/x86_64-linux-gnu/libicudata.so.66 (0x00007f34b4a6f000)
    libpcre.so.3 => /usr/lib/x86_64-linux-gnu/libpcre.so.3 (0x00007f34b49fc000)

 */

#include "mylib/include/mylib_global.h"
#include "mylib/include/mylib.h"

#include "include/first.h"

/**
 * https://doc.qt.io/qt-5/cmake-get-started.html
 * https://cmake.org/cmake/help/latest/guide/tutorial/Adding%20a%20Library.html
 * https://www.youtube.com/watch?v=l9CcfSRKeTM
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
