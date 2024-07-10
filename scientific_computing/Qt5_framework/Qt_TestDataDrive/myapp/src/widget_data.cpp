#include "../include/widget_data.h"

CWidgetData::CWidgetData(QObject *parent) : QObject(parent)
{
    qInfo() << this << "Constructor CWidgetdata ...";
}

void CWidgetData::testage_data()
{
    qInfo() << "Generating data...";

    // Qt's QTest::addColumn is a container that takes on 'QString' objects with paramter 'name'.
    // Qt's QTest::addColumn is a container that takes on 'int' objects with parameter 'age'.
    QTest::addColumn< QString >("name");
    QTest::addColumn< int >("age");

    // Qt's QTest::addRow is a function with paramter 'Invalid'
    QTest::addRow( "Invalid" ) << "Bob" << 190;
    QTest::addRow( "Old" ) << "Bryan" << 44;
    QTest::addRow( "Young" ) << "Heather" << 25;
    QTest::addRow( "Under age" ) << "Rango" << 14;
    QTest::addRow( "Retired" ) << "Grandma" << 90;

    qInfo() << "Data generated!";
}

void CWidgetData::testage()
{
    //Get the row data
    QFETCH(QString, name);
    QFETCH(int, age);

    qInfo() << "Testing age " << name << " is " << age;

    if(age < 1 || age > 100) 
        QFAIL("Invalid Age!");
    if(age < 21) 
        QFAIL("Must be an adult!");
    if(age > 40) 
        QWARN("Getting Old!");
    if(age > 65) 
        qInfo() << "This person is retired";

}
