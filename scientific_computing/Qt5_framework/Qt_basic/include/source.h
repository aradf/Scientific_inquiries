#ifndef SOURCE_HH                    // The compiler understand to define if it is not defined.
#define SOURCE_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

// Scientific Computational Library
namespace scl
{

class CSource : public QObject
{
    Q_OBJECT
public:
    explicit CSource(QObject * parent = 0);
    virtual ~CSource();

    void test();
signals:
    // A way to communicate to the destination to consume this signals.
    void my_signal(QString message);

public slots:    
};

} // end of Scientific Computational Library.

#endif 