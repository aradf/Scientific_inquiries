#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QDir>
#include <QBuffer>
#include <QTextStream>
#include <QDataStream>

#include <../include/first.h>
#include <../include/test.h>

QByteArray get_header() 
{
    // Specify Header
    QByteArray header;
    header.append("@!~!@");
    return header;
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

    {
        auto make_data = []() 
        {
            QByteArray data;

            for(int i = 0; i < 1000; i++) {
                data.append(QString::number(i) + "\r\n");
            }

            return data;
        };

        QByteArray data = make_data();
        QByteArray compressed = qCompress(data);
        QByteArray decompressed = qUncompress(compressed);

        qInfo() << "Original: " << data.length();
        qInfo() << "Compressed: " << compressed.length();
        qInfo() << "Decompressed: " << decompressed.length();
    }

    {
        QString orginal_file = QDir::currentPath() + QDir::separator() + "orginal.txt";
        QString compressed_file = QDir::currentPath() + QDir::separator() + "compressed.txt";
        QString decompressed_file = QDir::currentPath() + QDir::separator() + "decompressed.txt";

        //Our own custom file format, will not work with anything else!!
        auto make_file = [](QString path)
        {
            QFile file(path);
            if(file.open(QIODevice::WriteOnly)) 
            {
                QByteArray data;
                for(int i = 0; i < 1000; i++) 
                {            
                    data.append(QString::number(i) + "\r\n");
                }

                file.write(data);

                file.close();
                return true;
            }

            return false;
        };

        // function member 'compress_file' return a bool.  The parameters
        // are object type of Qt's QString class type.
        auto compress_file = [](QString originalFile, QString newFile)
        {
            // object 'original_file' is Qt's class type QFile.
            QFile original_file(originalFile);
            QFile new_file(newFile);

            // object 'header' is Qt's class type QByteArray.
            QByteArray header = get_header();

            // if the two files are not open for readonly and
            // writeonly exit.
            if(!original_file.open(QIODevice::ReadOnly)) 
                return false;
            if(!new_file.open(QIODevice::WriteOnly)) 
                return  false;
            
            // read buffer size of 1-k or 1024 bytes.
            const int size = 1024;

            // while not end of file continue.
            while (!original_file.atEnd()) 
            {
                // buffer is an object of Qt class type 'QByteArray'.
                // read 1024 bytes, compreress the 1024 bytes, print
                // postion pointer of the new file.  write the header
                // to the new file.  write the compress QByteArray to
                // new file.
                QByteArray buffer = original_file.read(size);
                QByteArray compressed = qCompress(buffer);
                qInfo() << "Header at:" << new_file.pos();
                new_file.write(header);
                qInfo() << "Size: " << new_file.write(compressed); // unknown size
            }
            original_file.close();
            new_file.close();
            return true;
        };

        auto decompress_file = [](QString originalFile, QString newFile)
        {
            QFile original_file(originalFile);
            QFile new_file(newFile);
            QByteArray header = get_header();
            const int size = 1024;

            if(!original_file.open(QIODevice::ReadOnly)) 
                return false;
            if(!new_file.open(QIODevice::WriteOnly)) 
                return false;

            // Double check that WE compressed the file
            // Note 'peek' reads 1024 bytes of the information 
            // looking for the header.  'peek' looks into the 
            // file however does not move the position
            // pointer 
            QByteArray buffer = original_file.peek(size);
            if(!buffer.startsWith(header)) 
            {
                qCritical() << "We did not create this file!";
                original_file.close();
                new_file.close();
                return false;
            }

            // Find the header positions
            // 'seek' reads into the file.  Unlike 'peek'
            // it moves the position pointer.  The position
            // pointer is right pass the first 'header'.
            original_file.seek(header.length());
            qInfo() << "Starting at: " << original_file.pos();

            while (!original_file.atEnd()) 
            {
                // 'peek' looks into the file.  unlike 'seek',
                // it does not move the position pointer. 
                buffer = original_file.peek(size);

                // find the position of next 'header' using
                // the indexOf command.
                qint64 index = buffer.indexOf(header);
                qInfo() << "Header found at:" << index;

                if(index > -1) 
                {
                    //Have the header in 1024 or 1-k of bytearray!
                    qint64 max_bytes = index;
                    qInfo() << "Reading: " << max_bytes;

                    // read move the position pointer to
                    // the beginning of 'header'.
                    buffer = original_file.read(max_bytes);
                    original_file.read(header.length());
                } else 
                {
                    //Do not have the header!
                    qInfo() << "Read all, no header";
                    buffer = original_file.readAll();
                }

                QByteArray decompressed = qUncompress(buffer);
                qInfo() << "Decompressed: " << new_file.write(decompressed);
                new_file.flush();
            }

            original_file.close();
            new_file.close();

            return true;
        };

        if(make_file(orginal_file)) {
            qInfo() << "Original created";

            if(compress_file(orginal_file, compressed_file)) {
                qInfo() << "File compressed";
                if(decompress_file(compressed_file,decompressed_file)) {
                    qInfo() << "File decompressed!";
                } else {
                    qInfo() << "Could not decompress file!";
                }
            } else {
                qInfo() << "File not compressed";
            }
        }
        qInfo() << "Data Compress and Decompress done ...";
    }

    {
        // Lambda Function.
        auto save_serializedFile = [](QString path)
        {
            QFile file(path);

            if(!file.open(QIODevice::WriteOnly))
                return false;
            
            QDataStream out(&file);
            out.setVersion(QDataStream::Qt_5_11);

            QString title = "The answer is 42";
            qint64 num = 42;

            out << title;
            out << num;

            file.flush();
            file.close();
        };

        if(save_serializedFile("test.txt"))
            qInfo() << "Saved ..";

        // read the serialized code back ...
        auto read_serializedFile = [](QString path)
        {
            QFile file(path);
            if(!file.open(QIODevice::ReadOnly))
                return false;
            
            QDataStream in(&file);
            if(in.version() != QDataStream::Qt_5_11)
            {
                qCritical() << "Bad Versin ...";
                file.close();
                return false;
            }
            QString title;
            qint64 num;

            in >> title;
            num >> num;

            qInfo() << title;
            qInfo() << num;

            return true;
        };

        if(read_serializedFile("test.txt"))
            qInfo() << "red file";

        qInfo() << "read/write serailized done ...";
    }

    {
        // Lambda Function: save_file returns an auto class type;
        // it takes a pointer variable of class type 'CTest';t == 0x1234
        // l-value; &t == 0xABCD; t is r-value;
        auto save_file = [](scl::CTest *t,QString path) 
        {
            QFile file(path);
            if(!file.open(QIODevice::WriteOnly)) return false;

            QDataStream ds(&file);
            ds.setVersion(QDataStream::Qt_5_11);

            //ds << t->name();
            // ds << t->map();

            ds << *t;

            file.close();
            return true;
        };


        auto load_file = [] (QString path) 
        {
            QFile file(path);
            if(!file.open(QIODevice::ReadOnly)) 
                return false;

            QDataStream ds(&file);

            scl::CTest t;
            ds >> t;

            file.close();

            qInfo() << "Name: " << t.name();
            foreach(QString key, t.map().keys()) 
            {
                qInfo() << key << " : " << t.map().value(key);
            }

            return true;
        };

        QString path = "test.txt";

        scl::CTest t;
        t.fill();
        
        if(save_file(&t,path)) 
        {
            load_file(path);
        }

        qInfo() << "read/write serailized done ...";
    }

    return a.exec();
}