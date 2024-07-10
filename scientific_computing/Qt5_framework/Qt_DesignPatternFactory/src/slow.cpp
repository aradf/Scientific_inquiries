#include "slow.h"

slow::slow(QObject *parent) : CCar(parent)
{

}


void slow::drive()
{
    qInfo() << "Max speed 80 mph";
}
