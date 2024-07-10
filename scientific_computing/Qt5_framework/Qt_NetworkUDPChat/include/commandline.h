#ifndef COMMANDLINE_H
#define COMMANDLINE_H

#include <QObject>
#include <QDebug>
#include <QTextStream>
#include <QtConcurrent>
#include <QThread>

// CCommandLine is child public child of QObject
class CCommandLine : public QObject
{
    // Gives access to the slot and signals
    Q_OBJECT
public:
    // explicit data conversion or casting.  file_handle is pointer object == 0x1234;
    // *file_handle is (r-value); Points to a memory location for content of FILE type
    // 
    explicit CCommandLine(QObject *parent = nullptr, FILE * file_handle = nullptr);

    // never returns loop for ever; must run on a differnt thread.
    [[noreturn]] void monitor();

signals:
    void command(QString value);

public slots:

private:
    QTextStream stream_input;

};

#endif // COMMANDLINE_H
