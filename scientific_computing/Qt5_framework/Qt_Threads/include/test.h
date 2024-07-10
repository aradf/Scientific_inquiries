#ifndef TEST_H
#define TEST_H

#include <QObject>           // Gives access to Q_OBJECT Macro
#include <QTimer>            // Gives access to Qtimer
#include <QDebug>            // Gives access to qInfo();
#include <QDateTime>         // Gives access to QDateTime
#include <QThread>           // Gives access to Qthreads

// Class CTest is a public child of QObject.
class CTest : public QObject
{
    Q_OBJECT
public:
    explicit CTest(QObject *parent = nullptr);
    ~CTest();

// emitted to trigger a behavior.
signals:

// slots consume an emitted signal
public slots:
    void timeout();
    void start();

private:
    QTimer timer;
};

#endif // TEST_H
