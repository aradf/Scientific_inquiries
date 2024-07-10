#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QDateTime>
#include <QIODevice>
#include <QTextStream>
#include <QLoggingCategory>
#include <QRandomGenerator>
#include <QTextCodec>

#include <../include/first.h>

// Dclare a logging category
Q_DECLARE_LOGGING_CATEGORY(network);
Q_LOGGING_CATEGORY(network, "network");

const QtMessageHandler QT_DEFAULT_MESSAG_HANDLER = qInstallMessageHandler(nullptr);

void log_handler(QtMsgType type, const QMessageLogContext & context, const QString &msg)
{
    // qInfo() << "log_handler is invoked ...";
    QString text_string;
    switch(type)
    {
        case QtInfoMsg:
                text_string = QString("Info: %1 in %2").arg(msg);
                break;
        case QtDebugMsg:
            text_string = QString("Debug: %1").arg(msg);
            break;
        case QtWarningMsg:
            text_string = QString("Warning: %1").arg(msg);
            break;
        case QtCriticalMsg:
            text_string = QString("Critical: %1").arg(msg);
            break;
        case QtFatalMsg:
            text_string = QString("Fatal: %1").arg(msg);
            break;
    }
    QFile file("log.txt");
    if (file.open(QIODevice::WriteOnly | QIODevice::Append))
    {
        QTextStream ts(&file);
        ts << QDateTime::currentDateTime().toString() << "-" 
                                                      << text_string 
                                                      << "file: " 
                                                      << context.file 
                                                      << "Line: ";
        ts.flush();
        file.close();
    }

    (*QT_DEFAULT_MESSAG_HANDLER)(type, context, msg);
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

    qInstallMessageHandler(log_handler);

    {
        qInfo() << "This is a info message";
        qDebug() << "This is a debug message";
        qWarning() << "This is a warning message";
        qCritical() << "This is a critical message";
        // qFatal() << "The end ...";
    }

    {
        qInfo() << "test";
        qInfo(network)  << "test 1";
        QLoggingCategory::setFilterRules("network.debug=false");
        qInfo(network)  << "test 2";
        QLoggingCategory::setFilterRules("network.debug=true");
        qInfo(network)  << "test 3";
    }

    {
        // ASCII is a 7-bit character set cotaining 128 characters.
        // It contains the numbers from 0-9, the upper and lower case
        // English letters from A to Z, and some special characters.
        // The character sets used in the modern computers, the HTML, and
        // on the internet, are all based on ASCII.
        // www.asciitable.com
        // American Standard Code for Information Interchange.

        // Lambda Function ...
        auto make_data = []()
                         { 
                            QByteArray data; 
                            for(int i = 0; i < 10; i++) 
                            {
                                data.append(72);    // H
                                data.append(101);   // e
                                data.append(108);   // l
                                data.append(108);   // l
                                data.append(111);   // o
                                data.append(33);    // 
                                data.append(13); // \r
                                data.append(10); // \n
                            }
                            return data;
                         };

        QByteArray data = make_data();
        qInfo() << data;

        QString path = QDir::currentPath() + QDir::separator() + "data.txt";
        QFile file(path);

        if(file.open(QIODevice::WriteOnly)) 
        {
            file.write(data);
            file.close();
        }        
        qInfo() << "Completed ...";
    }

    {
        /*

        UTF-8 is a variable width character encoding capable of encoding all 1,112,064 valid code points in
        Unicode using one to four 8-bit bytes.
        The encoding is defined by the Unicode standard, and was originally designed by Ken Thompson and Rob Pike.
        Called "Unicode"
        */
         // Lambda Function ...
        auto make_data = []()
                         { 
                            QString data; 
                            data.append("Unicode test\r\n");
                            for(int i = 0; i < 10; i++)
                            {
                                int number = QRandomGenerator::global()->bounded(1112064);
                                data.append(number);
                            }
                            data.append("\r\n");
                            return data;
                         };

        QString data = make_data();
        QString path = QDir::currentPath() + QDir::separator() + "data.txt";
        QFile file(path);
        if(file.open(QIODevice::WriteOnly))
        {
            QTextStream stream(&file);
            stream.setCodec("UTIF-8");
            stream << data;
            stream.flush();
            file.close();
        }

        qInfo() << "Done";
        qInfo() << "Unicode: " << data;
        qInfo() << " ";
        qInfo() << "ASCII: " << data.toLatin1();

        qInfo() << "Completed ...";
    }

    {
        /*
         * Bae 64 is a group of similar binary to text encoding schemes;  They represent binary 
         * data in an ASCIIformat.  Each Base64 digit represents exactly 6 bits of data.
         * https://www.base64decode.org/
         */
        auto make_data = []()
        {
            QString data;
            for (int i=0; i<10; i++)
            {
                data.append("Hello\r\n");
            }
            return data;
        };
        
        QString data = make_data();
        QByteArray bytes(data.toLatin1());
        QByteArray encoded = bytes.toBase64();
        QByteArray decoded = QByteArray::fromBase64(encoded);

        qInfo() << "Encoded ...." << encoded;
        qInfo() << "-------------------";
        qInfo() << "Decoded ...." << decoded;

        qInfo() << "Completed ...";
    }

    {
        /*
         * Hex
         * https://www.base64decode.org/
         */
        auto make_data = []()
        {
            QString data;
            for (int i=0; i<10; i++)
            {
                data.append("Hello\r\n");
            }
            return data;
        };
        
        QString data = make_data();
        QByteArray bytes(data.toLatin1());
        QByteArray encoded = bytes.toHex();
        QByteArray decoded = QByteArray::fromHex(encoded);

        qInfo() << "Hex: Encoded ...." << encoded;
        qInfo() << "-------------------";
        qInfo() << "Decoded ...." << decoded;
        qInfo() << "Completed ...";
    }

    {
        /*
         * 
         * https://www.base64decode.org/
         */
        auto make_data = []()
        {
            QString data;
            for (int i=0; i<10; i++)
            {
                data.append("Hello\r\n");
            }
            return data;
        };
        
        QString data = make_data();
        QTextCodec *codec = QTextCodec::codecForName("UTF-16");
        if(!codec) qFatal("No codec!");

        QByteArray bytes = codec->fromUnicode(data);
        qInfo() << "Bytes: " << bytes;
        qInfo() << "-----------------------------------------";
        QString string = codec->toUnicode(bytes);
        qInfo() << "String: " << string;

        qInfo() << "Completed ...";
    }

    return a.exec();
}