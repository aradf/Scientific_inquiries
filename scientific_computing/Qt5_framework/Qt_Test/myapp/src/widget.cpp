#include "../include/widget.h"

scl::CWidget::CWidget(QObject *parent) : QObject(parent)
{
    age = 0;
}

void scl::CWidget::set_age(int value)
{
    age = value;
}

void scl::CWidget::test_fail()
{
    QFAIL("NO REASON JUST FAIL!!!");
}

void scl::CWidget::test_age()
{
    if(!age) 
        QFAIL("Age is not set!");
}

void scl::CWidget::test_widget()
{

    int value = 45;

    //Make sure the age is valid
    QVERIFY(age > 0 && age < 100);

    //Issue warnings
    if(age > 40) 
        QWARN("Age is over 40!");

    if(age < 21) 
        QFAIL("Must be an adult!");

    //Make sure they are the same
    QCOMPARE(age, value);


}
