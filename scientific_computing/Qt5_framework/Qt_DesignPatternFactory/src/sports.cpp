#include "sports.h"

sports::sports(QObject *parent) : CCar(parent)
{

}


void sports::drive()
{
    qInfo() << "Max speed 120 mph";
}
