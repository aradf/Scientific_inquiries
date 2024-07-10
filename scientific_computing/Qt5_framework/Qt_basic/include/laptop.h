#ifndef LAPTOP_H
#define LAPTOP_H

#include <QObject>              // Give access to slots and signals
#include <QDebug>               // Give access to qDebug() and qInfo()

namespace scl
{

class CLaptop : public QObject
{
    Q_OBJECT
public:
    explicit CLaptop(QObject * parent = nullptr);
    explicit CLaptop(QObject * parent = nullptr, 
                     QString some_name = "");
    ~CLaptop();

    double as_kilograms();
    void test();

    int weight_;
    QString name_;

signals:

public slots:

};

} // scl namespace

#endif