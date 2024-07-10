#include <QCoreApplication>
#include <QDebug>
#include <QList>
#include <QVector>
#include <QtAlgorithms>
#include <QRandomGenerator>
#include <QSysInfo>
#include <QProcess>

#include "../include/test.h"
#include "../include/commander.h"

// User-define custom
#define add(a,b ) a+b

// random function returns void; takes a pointer object;
// The Qt's class QVector is a template class holding inters;
// list == 0x1234 (l-value); &list == 0xABCD; list is instance
// of the temp-data (r-value)
void randoms(QVector<int> *ptr_list, int max)
{
    ptr_list->reserve(max);
    for (int i = 0 ;i < max ; i++)
    {
        int value = QRandomGenerator::global()->bounded(1000);
        ptr_list->append(value);
    }
}

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
        // qDeleteAll
        // Qt's QList class holds pointers to the CTest class;
        // list of pointers;
        QList<CTest*> ctest_list;
        
        for(int i = 0; i< 10; i++) 
        {
            // pointer object 
            CTest *ptr_test = new CTest(); //notice no parent!!!
            ptr_test->setObjectName(QString::number(i));
            ctest_list.append(ptr_test);
        }

        qInfo() << ctest_list.at(0);

        //Delete them all!
        qDeleteAll(ctest_list);

        //Will crash!
        //qInfo() << list.at(0);

        //Objects are still in the list but invalid pointers - clear the list!
        ctest_list.clear();
    }

    {
        QVector<int> list;
        list << 1 << 2 <<3 << 4;
        qInfo() << list;

        // qFill(list, 0);       // Obsolete code, still works.
         
        list.fill(0);
        qInfo() << list;
    }

    {
        // qSort
        // Embedded Class Functor;
        class CFunctor
        {
        private:
            int number_;
        public:
            void operator()(QString some_string)
            {
                qInfo() << "Calling functor CFunctor with parameters: " << some_string;
            }

            // type conversion function: casting an instance of X with std::string
            operator QString () const 
            { 
                return "CFunctor";
            }

            // Constructor for CFunctor
            CFunctor(int i) 
            {
                qInfo() << "Init object of class CFunctor " << i; 
                this->number_ = i;
            }

            // Constructor for CFunctor
            CFunctor() 
            {
                qInfo() << "Init object of class CFunctor "; 
                this->number_ = 0;
            }
        };

        CFunctor(8)("Hello_world!");

        // Qt's QVector template class holding integers,
        // list is pointer variable;
        QVector<int> list;

        // pass the pointer variable to the randoms function
        // using it's referance or by address &list == 0xABCD
        randoms(&list, 10);
        qInfo() << "Unsorted " << list;

        qSort(list);
        qInfo() << "Sorted: " << list;

        std::sort(list.begin(), list.end());
        qInfo() << "Sorted: " << list;

        qInfo() << "Life is good ...";
    }

    {
        // User defined macros
        qInfo() << add(1, 2);

        // Lets break it;
        // BAD BAD BAD
        qInfo() << add("Test",22);
        qInfo() << add(true,true);
        qInfo() << add(false, true);
        qInfo() << add(false, false);
        qInfo() << add("Test",'\n');
        // qInfo() << add("Test",'/n');
        
        qInfo() << "Life is good ...";
    }

    {
        // QSysInfo
        QSysInfo sys;
        qInfo() << "Boot Id: " << sys.bootUniqueId();
        qInfo() << "Build: " << sys.buildAbi();
        qInfo() << "Cpu: " << sys.buildCpuArchitecture();
        qInfo() << "Kernel: " << sys.kernelType();
        qInfo() << "Version: " << sys.kernelVersion();
        qInfo() << "Mac: " << sys.macVersion();
        qInfo() << "Windows: " << sys.windowsVersion();
        qInfo() << "Host: " << sys.machineHostName();
        qInfo() << "Product: " << sys.prettyProductName();
        qInfo() << "Type: " << sys.productType();
        qInfo() << "Version: " << sys.productVersion();

    #ifdef Q_OS_LINUX
        qInfo() << "Linux code here";
    #elif defined(Q_OS_WIN32)
        qInfo() << "Windows code here";
    #elif defined(Q_OS_MACX)
        qInfo() << "Mac code here";
    #else
        qInfo() << "Unknown OS code here";
    #endif   

        qInfo() << "Life is good ...";
    }

    {
        // QProcess: 
        auto test_process = []() 
        {
            QProcess gzip;
            gzip.start("gzip", QStringList() << "-c");

            if(!gzip.waitForStarted()) 
                return false;

            gzip.write("Qt rocks!");
            gzip.closeWriteChannel();

            if(!gzip.waitForFinished()) 
                return false;

            QByteArray result = gzip.readAll();
            qInfo() << "Result: " << result;

            return true;

        };

        if(test_process()) 
        {
            qInfo() << "Compressed with gzip!";
        } else 
        {
            qInfo() << "Failed to use gzip";
        }

        qInfo() << "Life is good ...";
    }

    {
        // QProcess: 
        qInfo() << "Starting ...";
        QProcess proc;
        proc.execute("xed", QStringList() << "http://www.nfl.com");
        qInfo() << "Exit Info: " << proc.exitCode();

        qInfo() << "Life is good ...";
    }

    {
        CCommander cmd;
        cmd.action(QByteArray("pwd"));
        cmd.action(QByteArray("ls -l"));

        // system("pwd")
        // system("ls -l")
        qInfo() << "Life is good ...";
    }

    return a.exec();
}
