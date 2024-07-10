#ifndef WIDGET_H
#define WIDGET_H

#include <QObject>
#include <QDebug>
#include <QTest>

namespace scl
{

class CWidget : public QObject
{
    Q_OBJECT
public:
    explicit CWidget(QObject *parent = nullptr);

    void set_age(int value);

signals:

public slots:

private slots:
    void test_fail();
    void test_age();
    void test_widget();

private:
    int age;
};

}

#endif // WIDGET_H
