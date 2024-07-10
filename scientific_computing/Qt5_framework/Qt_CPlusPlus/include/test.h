#ifndef TEST_HH                    // The compiler understand to define if it is not defined.
#define TEST_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()
#include <QPointer>

#include <string.h>                  // gives access to strcpy.

// Scientific Computational Library
namespace scl
{
class CTest : public QObject
{
    Q_OBJECT
public:
    explicit CTest(QObject * parent = 0);
    virtual ~CTest();

    //QString some_string_;
    char name_[50];

    // Qt's templated class QPointer has a class type QObject
    // object variable widget_;
    QPointer<QObject> widget_;

    void make_child(QString name);
    void use_widget();
    void print_name();

signals:

public slots:    
};

} // end of Scientific Computational Library.

#endif 