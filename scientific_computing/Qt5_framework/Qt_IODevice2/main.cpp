#include <QCoreApplication>
#include <QIODevice>
#include <QDir>
#include <QFile>
#include <QString>
#include <QByteArray>
#include <QDebug>
#include <QTextStream>
#include <QDataStream>
#include <QFileInfo>
#include <QRandomGenerator>
#include <QLockFile>

#include <../include/first.h>

bool create_textFile(QString path) {
    QFile file(path);
    if(!file.open(QIODevice::WriteOnly)) {
        qWarning() << file.errorString();
        return false;
    }

    QTextStream stream(&file);
    int max = 1000;
    QString banner = "Random number:\r\n";
    qInfo() << "Writing " << banner;
    stream << banner;

    for(int i = 0; i < 5; i++) {
        qint32 num = QRandomGenerator::global()->bounded(max);
        qInfo() << "Writing: " << num;
        stream << num << "\r\n";
    }

    file.close();
    return true;
}

void read_textFile(QString path) {
    QFile file(path);
    if(!file.open(QIODevice::ReadOnly)) {
        qWarning() << file.errorString();
        return;
    }

    QTextStream stream(&file);

    QString banner;
    stream >> banner;

    qInfo() << "Banner: " << banner;

    while (!stream.atEnd()) {
        //qint32 num;
        QString num;
        stream >> num;
        if(!num.isEmpty()) qInfo() << "Random: " << num;
    }

    /*
    for(int i = 0; i < 5; i++) {
       //qint32 num;
       QString num;
       stream >> num;
       qInfo() << "Random: " << num;

    }
     */

    file.close();
}

bool write(QString path, QByteArray data) {
    QFile file(path);
    if(!file.open(QIODevice::WriteOnly)) {
        qWarning() << file.errorString();
        return false;
    }

    qint64 bytes = file.write(data);
    file.close();
    if(bytes) return true;

    return false;
}

bool createfile(QString path) {
    QByteArray data;
    for(int i = 0; i < 5; i++) {
        data.append(QString::number(i));
        data.append(" Hello World\r\n");
    }

    return write(path,data);
}

void readFile(QString path) {
    QFile file(path);
    if(!file.exists()) {
        qWarning() << "File not found";
        return;
    }

    if(!file.open(QIODevice::ReadOnly)) {
        qWarning() << file.errorString();
        return;
    }

    qInfo() << "**** Reading File";
    qInfo() << file.readAll(); //Best small files!
    qInfo() << "**** Done";

    file.close();
}

void readLines(QString path) {
    QFile file(path);
    if(!file.exists()) {
        qWarning() << "File not found";
        return;
    }

    if(!file.open(QIODevice::ReadOnly)) {
        qWarning() << file.errorString();
        return;
    }

    qInfo() << "**** Reading File";
    while (!file.atEnd()) {
        QString line(file.readLine());
        qInfo() << "Read:" << line.trimmed(); //best with text files!
    }
    qInfo() << "**** Done";

    file.close();
}

void readBytes(QString path) {
    QFile file(path);
    if(!file.exists()) {
        qWarning() << "File not found";
        return;
    }

    if(!file.open(QIODevice::ReadOnly)) {
        qWarning() << file.errorString();
        return;
    }

    qInfo() << "**** Reading File";
    while (!file.atEnd()) {
        qInfo() << "Read: " << file.read(5); // best with larger file or structs
    }
    qInfo() << "**** Done";

    file.close();
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

    QString path = QDir::currentPath() + QDir::separator() + "test.txt";
    qInfo() << "Path: " << path;

    if(createfile(path)) {
        readFile(path);
        readLines(path);
        readBytes(path);
    }

    {
        // QDataStream 
        auto create_file = [](QString path)
             {
                QFile file(path);
                if(!file.open(QIODevice::WriteOnly))
                {
                    qWarning() << file.errorString();
                    return false;
                }
                QDataStream stream(&file);
                int max = 1000;
                QString banner = "Random Numbers:";
                qInfo() << "Writing " << banner;
                stream << banner;

                for(int i = 0; i < 5; i++)
                {
                    qint32 value = QRandomGenerator::global()->generate();
                    qint32 num = QRandomGenerator::global()->bounded(max);
                    qInfo() << "Writing: " << num;
                    stream << num;
                }

                file.flush();
                file.close();
                return true;
             };

        auto read_file = [](QString path)
             {
                QFile file(path);
                if(!file.open(QIODevice::ReadOnly))
                {
                    qWarning() << file.errorString();
                    return ;
                }
                QDataStream stream(&file);
                QString banner;
                stream >> banner;

                qInfo() << "Banner" << banner;
                for(int i = 0; i < 5; i++)
                {
                    qint32 num;
                    stream >> num;
                    qInfo() << "Reading: " << num;
                    stream << num;
                }
             };

        QString path = QDir::currentPath() + QDir::separator() + "test.txt";
        qInfo() << "path: " << path;
        if(create_file(path))
        {
            read_file(path);
        }
    }

    {
        QString path = QDir::currentPath() + QDir::separator() + "test.txt";
        qInfo() << "Path: " << path;
        if(create_textFile(path)) 
        {
            read_textFile(path);
        }
    }

    {
        // Lock a file.
        QString path = QDir::currentPath() + QDir::separator() + "test.txt";
        QFile file(path);
        QLockFile lock(file.fileName() +"l");
        lock.setStaleLockTime(30000);

        if(lock.tryLock()) 
        {
            qInfo() << "Putting into file...";
            if(file.open(QIODevice::WriteOnly)) {
                QByteArray data;
                file.write(data);
                file.close();
                qInfo() << "Finished with file...";
                //Took over 30 seconds here, auto unlock
            }
            qInfo() << "Unlocking file";
            lock.unlock();
        } 
        else 
        {
            qInfo() << "Could not lock the file!";
            qint64 pid;
            QString host;
            QString application;

            if(lock.getLockInfo(&pid,&host,&application)) 
            {
                qInfo() << "The file is locked by:";
                qInfo() << "Pid: " << pid;
                qInfo() << "Host: " << host;
                qInfo() << "Application: " << application;

            } 
            else 
                {
                    qInfo() << "File is locked, but we can't get the info!";
                }
        }
    }

    return a.exec();
}
