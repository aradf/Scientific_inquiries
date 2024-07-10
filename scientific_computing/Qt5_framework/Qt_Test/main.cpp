#include <QCoreApplication>
#include <QDebug>
#include <QTest>

#include "myapp/include/first.h"
#include "myapp/include/cat.h"
#include "myapp/include/dog.h"
#include "myapp/include/widget.h"

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
        scl::CCat my_cat;
        QTest::qExec(&my_cat);

        scl::CDog my_dog;
        QTest::qExec(&my_dog);

        scl::CWidget widget;
        widget.set_age(20);
        QTest::qExec(&widget);
    }

    return a.exec();
}
