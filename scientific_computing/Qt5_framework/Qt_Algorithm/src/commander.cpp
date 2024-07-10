#include "../include/commander.h"

CCommander::CCommander(QObject *parent) : QObject(parent)
{
#ifdef Q_OS_WIN
    //Windows
    app = "cmd";
    endl = "\r\n";
#else
    //Not windows
    app = "bash";
    endl = "\n";
#endif

    proc.setProgram(app);
    proc.start();
    proc.setReadChannelMode(QProcess::MergedChannels);

    // connect the referance/address of this classe's member function 'proc' to
    // the QProcess's signal 'readyRead'. The CCommander's ready_read slot. 
    connect(&proc,&QProcess::readyRead, this, &CCommander::ready_read);
    connect(&proc,&QProcess::readyReadStandardOutput, this, &CCommander::ready_read);
    connect(&proc,&QProcess::readyReadStandardError, this, &CCommander::ready_read);
}

CCommander::~CCommander()
{
    if(proc.isOpen()) 
        proc.close();
}

void CCommander::ready_read()
{
    qint64 value = 0;
    do 
    {
        QByteArray line = proc.readAll();
        qInfo() << line.trimmed();
        value = line.length();
    }while(value > 0);
}

void CCommander::action(QByteArray data)
{
    if(!data.endsWith(endl.toLatin1())) 
        data.append(endl);

    proc.write(data);
    proc.waitForBytesWritten(1000);
}
