#ifndef WATCHER_HH                    // The compiler understand to define if it is not defined.
#define WATCHER_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CWatcher : public QObject
{
    Q_OBJECT
private:
    QString message_;

public:
    explicit CWatcher(QObject * parent = 0);
    virtual ~CWatcher();
    
    QString return_message();
    void set_message(QString some_value);
signals:

// Consume a signal
public slots:    
    void message_changed(QString message);
};

} // end of Scientific Computational Library.

#endif 