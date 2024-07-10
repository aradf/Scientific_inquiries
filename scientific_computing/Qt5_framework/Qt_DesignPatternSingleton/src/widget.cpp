#include "widget.h"

widget::widget(QObject *parent) : QObject(parent)
{

}

void widget::make_changes(QString value)
{
    CTest::instance()->name_ = value;
}
