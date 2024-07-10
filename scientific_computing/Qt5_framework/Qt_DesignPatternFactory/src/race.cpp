#include "race.h"

race::race(QObject *parent) : CCar(parent)
{

}

void race::drive()
{
    qInfo() << "Max speed 200 mph";
}
