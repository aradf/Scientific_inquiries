#ifndef TEST_H
#define TEST_H

#include <QObject>

class CTest : public QObject
{
    Q_OBJECT
public:
    explicit CTest(QObject *parent = nullptr);

signals:

public slots:
};

#endif // TEST_H
