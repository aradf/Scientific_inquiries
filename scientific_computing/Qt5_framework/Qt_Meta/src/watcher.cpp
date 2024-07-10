#include "watcher.h"

CWatcher::CWatcher(QObject *parent) : QObject(parent)
{
    file_systemWatcher.addPath(QDir::currentPath());
    file_systemWatcher.addPath(QDir::currentPath() + QDir::separator() + "test.txt");

    QObject::connect(&file_systemWatcher,&QFileSystemWatcher::fileChanged, this, &CWatcher::fileChanged);
    QObject::connect(&file_systemWatcher,&QFileSystemWatcher::directoryChanged, this, &CWatcher::directoryChanged);
}

void CWatcher::fileChanged(const QString &path)
{
    qInfo() << "File changed: " << path;
    if(file_systemWatcher.files().length()) 
    {
        qInfo() << "Files that have changed:";
        foreach(QString file, file_systemWatcher.files()) 
        {
            qInfo() << file;
        }
    }
}

void CWatcher::directoryChanged(const QString &path)
{
    qInfo() << "Directory changed: " << path;
    if(file_systemWatcher.directories().length()) 
    {
        qInfo() << "Dir changed:";
        foreach(QString dir, file_systemWatcher.directories()) 
        {
            qInfo() << dir;
        }
    }
}
