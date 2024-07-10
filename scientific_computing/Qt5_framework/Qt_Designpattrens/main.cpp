#include <QCoreApplication>
#include <QDebug>

#include <../include/first.h>
#include <../include/pool.h>

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


        CPool ptr_pool;
        for(int i = 0; i < 100; i++) 
        {
            ptr_pool.work(i);
        }

    }

    return a.exec();
}
