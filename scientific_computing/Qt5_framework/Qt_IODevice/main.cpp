#include <QtCore/QCoreApplication>   // Gives access to QCoreApplication.
#include <QDebug>                    // Gives access to qDebug, qInfo.
#include <QProcessEnvironment>       // Gives access to QProcessEnvironment
#include <QIODevice>
#include <QBuffer>
#include <QDir>
#include <QString>
#include <QFileInfo>
#include <QFileInfoList>
#include <QDateTime>
#include <QStorageInfo>
#include <QFile>
#include <QByteArray>

#include <../include/first.h>

bool put (QString path, QByteArray data, QIODevice::OpenMode mode)
{
    QFile file(path);
    if(!file.open(mode))
    {
        qWarning() << file.errorString();
        return false;
    }
    qint64 bytes = file.write(data);
    if(!bytes)
    {
        qWarning() << file.errorString();
    }
    else
        {
            qInfo() << "Wrote " << bytes;
        }
    file.flush();
    file.close();

    return true;
};

/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */
int main (int argc, char * argv[])
{
    QCoreApplication a(argc, argv);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();

    QString path = env.value("Path");
    qDebug() << "Path Values: " 
             << path;

    {
        QBuffer buffer;
        if(buffer.open(QIODevice::ReadWrite))
        {
            qInfo() << "Buffer Openned ...";
            QByteArray data("hello World!");

            for(int i = 0; i < 5; i++)
            {
                buffer.write(data);
                buffer.write("\r\n");
            }
            // File and device access you may need to flush the data to the device.
            // buffer.flush();

            // Move to the first position.
            buffer.seek(0);

            qInfo() << buffer.readAll();
            qInfo() << "Closing the buffer ...";

            buffer.close();
        }
        else
            qInfo() << "Buffer Open Failed ...";


        qInfo() << "Done";        
    }

    {
        qInfo() << "\n\n";
        // Lambda Function create_dir returns bool; pointer object to Qt's 'QString'
        // path is an object (r-value); 
        auto create_dir = []
                        (QString path)
                        {
                            qInfo() << path;
                            QDir dir(path);
                            if (dir.exists())
                            {
                                qInfo() << "Already Exist" << path;    
                                return true;
                            }
                            
                            if(dir.mkpath(path))
                            {
                                qInfo() << "Created" <<path;    
                                return true;
                            }

                            return false;                          
                        };

        auto rename  =  []
                        (QString path, QString name)
                        {
                            QDir dir(path);

                            if (!dir.exists())
                            {
                                qInfo() << "Path does not exist ...";
                                return false;
                            }
                            //Linux is // and windows is \\.
                            int pos = path.lastIndexOf(QDir::separator());
                            QString parent = path.mid(0,pos);
                            QString newpath = parent = QDir::separator() + name;

                            qInfo() << "Absolute: " << dir.absolutePath();
                            qInfo() << "Parent: " << parent;
                            qInfo() << "New: " << newpath;

                            bool flag = dir.rename(path,newpath);
                            return flag;                          
                        };

        auto remove  =  []
                        (QString path)
                        {
                            qInfo() << "Remove: " << path;
                            QDir dir(path);
                            if (!dir.exists())
                            {
                                qInfo() << "Path does not exist ...";
                                return false;
                            }

                            //DANGER;
                            // bool flag removeRecursively();

                            return true;                          
                        };

        QString path = QDir::currentPath();
        QString test = path + QDir::separator() + "test";
        QString tmp  = path + QDir::separator() + "tmp";

        bool flag = false;
        qInfo() << test;

        // The next few lines work; commented out for safety.        
        // flag = create_dir(test);
        // flag = rename(test,tmp);
        // flag = remove(tmp);

        QDir current(QDir::currentPath());
        if (current.exists())
        {
            foreach(QFileInfo fi, current.entryInfoList())
            {
                qInfo() << fi.fileName();
            }            
        }

        qInfo() << "Done";        
    }

    {
        qInfo() << "\n\n";
        // Lambda Function create_dir returns bool; pointer object to Qt's 'QString'
        // path is an object (r-value); 
        auto list_dir = []
                        (QString path)
                        {
                            qInfo() << path;

                            QDir dir(path);
                            qInfo() << dir.absolutePath();

                            QFileInfoList dirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
                            QFileInfoList files = dir.entryInfoList(QDir::Files);

                            foreach(QFileInfo fi, dirs)
                            {
                                qInfo() << fi.fileName();
                            }

                            foreach(QFileInfo fi, files)
                            {
                                qInfo() << fi.fileName();
                                qInfo() << fi.size();
                                qInfo() << fi.birthTime();
                                qInfo() << fi.lastModified();
                            }

                            // Could not make it recursive.
                            // foreach(QFileInfo f1, dirs)
                            // {
                            //     list_dir(fi.absoluteFilePath());
                            // }

                            return;  
                        };

        QString path = QDir::tempPath();
        qInfo() << path;
        list_dir(path);

        qInfo() << "Done";        
    }

    {
        qInfo() << "\n\n";
        foreach(QStorageInfo storage_item, QStorageInfo::mountedVolumes())
        {
            qInfo() << "Name: " << storage_item.displayName();
            qInfo() << "Type: " << storage_item.fileSystemType();
            qInfo() << "Total: " << storage_item.bytesTotal()/(1000*1000) << "MB";            
            qInfo() << "Available: " << storage_item.bytesAvailable()/(1000*1000) << "MB";
            qInfo() << "Device: " << storage_item.device();
            qInfo() << "Device: " << storage_item.isRoot();            
        }
        QStorageInfo root = QStorageInfo::root();
        QDir dir(root.rootPath());
        foreach(QFileInfo item, dir.entryInfoList(QDir::Dirs | QDir::NoDotDot))
        {
            qInfo() << item << " " << item.filePath();
        }

        qInfo() << "Done";        
    }

    {
        qInfo() << "\n\n";

        auto write =  [] (QString path, QByteArray data)
                      {
                        qInfo() << "Write to the file ";
                        bool flag = put(path, data, QIODevice::WriteOnly);
                        if(flag)
                        {
                            qInfo() << "Data written to file";
                        }
                        else
                            {
                                qWarning() << "Failed to write to file";
                            }
                      };

        auto append =  [] (QString path, QByteArray data)
                      {
                        qInfo() << "Write to the file ";
                        bool flag = put(path, data, QIODevice::Append);
                        if(flag)
                        {
                            qInfo() << "Data written to file";
                        }
                        else
                            {
                                qWarning() << "Failed to write to file";
                            }
                      };

        QString path = QDir::currentPath() + QDir::separator() + "test.txt";
        qInfo() << path;
        
        QByteArray data("Hello World!\n");

        for(int i = 0; i < 5; i++)
        {
            QString num = QString::number(i);
            QByteArray line(num.toLatin1() + data);
            write(path, line);
        }

        qInfo() << "Done";        
    }


    return a.exec();
}
