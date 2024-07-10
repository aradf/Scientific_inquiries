#ifndef TEST_H
#define TEST_H

#include <QObject>
#include "singleton.h"

class CTest : public QObject
{
    Q_OBJECT
    static CTest *createInstance();
    // Static function returning a pointer to the CTest class.

public:
    explicit CTest(QObject *parent = nullptr);

    QString name_;
    // Static function returning a pointer to the CTest class.
    static CTest* instance();

signals:

public slots:
};

#endif // TEST_H
