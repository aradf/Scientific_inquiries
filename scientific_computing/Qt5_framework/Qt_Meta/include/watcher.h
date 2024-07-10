#ifndef WATCHER_H
#define WATCHER_H

#include <QObject>
#include <QDebug>
#include <QDir>
#include <QFileSystemWatcher>

class CWatcher : public QObject
{
    Q_OBJECT
public:
    explicit CWatcher(QObject *parent = nullptr);

signals:

public slots:
    void fileChanged(const QString &path);
    void directoryChanged(const QString &path);

private:
    QFileSystemWatcher file_systemWatcher;
};

#endif // CWatcher_H
