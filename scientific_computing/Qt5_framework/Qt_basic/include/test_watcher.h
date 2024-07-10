#ifndef TEST_WATCHER_HH                    // The compiler understand to define if it is not defined.
#define TEST_WATCHER_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CTestwatcher : public QObject
{
    Q_OBJECT

private:
    QString message_;
public:
    explicit CTestwatcher(QObject * parent = 0);
    virtual ~CTestwatcher();
    
    /*
        Data Type: QString
        Name: message
        READ: Where to Read from.
        message: message function member.
        WRITE: What to use when one wants to WRITE the value.
        set_message: set_message function member.
        NOTIFY: When this is changed and you want to be notifed this is 
                what you need to connect to.
        message_changed: message_changed function member.

        Q_PROPERTY is a macro, so there is no need for ';'
    */
    Q_PROPERTY(QString message READ message WRITE set_message NOTIFY message_changed)

    QString message();
    void set_message(QString value);
// signal sends a message to the consuming slot.
signals:
    void message_changed(QString message);

public slots:    
};


} // end of Scientific Computational Library.

#endif 