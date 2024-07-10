#ifndef MYLIB_H
#define MYLIB_H

#include <QDebug>
#include <QString>
#include "mylib_global.h"

class MYLIBSHARED_EXPORT Mylib
{

public:
    Mylib();
    ~Mylib();
    void test_library();
    QString get_name();
};

#endif // MYLIB_H
