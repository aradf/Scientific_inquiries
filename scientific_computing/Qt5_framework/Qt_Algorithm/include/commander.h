#ifndef COMMANDER_H
#define COMMANDER_H

#include <QObject>
#include <QDebug>
#include <QProcess>

class CCommander : public QObject
{
    Q_OBJECT
public:
    explicit CCommander(QObject *parent = nullptr);
    ~CCommander();
signals:

public slots:
    void ready_read();
    void action(QByteArray data);

private:
    QProcess proc;
    QString app;
    QString endl;
};

#endif // COMMANDER_H
