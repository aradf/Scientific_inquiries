#include "thread_counter.h"

CThreadCounter::CThreadCounter(QObject *parent) : QObject(parent)
{

}

void CThreadCounter::increment()
{
    this->value++;
}
void CThreadCounter::decrement()
{
    this->value--;
}

int CThreadCounter::return_count()
{
    return (this->value);
}

