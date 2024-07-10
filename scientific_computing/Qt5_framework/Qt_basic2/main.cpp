#include <QtCore/QCoreApplication>   // Gives access to QCoreApplication.
#include <QDebug>                    // Gives access to qDebug, qInfo.
#include <QProcessEnvironment>       // Gives access to QProcessEnvironment
#include <QDateTime>
#include <QTime>
#include <QDate>
#include <QVariant>

#include <array>                     // array from 'STL'
#include <iostream>                  // Gives access to std::cout.

#include <../include/animal.h>
#include <../include/car.h>
#include <../include/feline.h>
#include <../include/racecar.h>
#include <../include/dog.h>
#include <../include/test.h> 

void test_drive(scl::CCar * ptr_car)
{
    ptr_car->drive();
    ptr_car->stop();
}

void test_raceCar(scl::CRacecar * ptr_raceCar)
{
    ptr_raceCar->go_fast();
}

bool doDivision(int max) {
    try {
        int value;
        qInfo() << "Enter a number";
        value = 1;;

        if(value == 0) throw "Can not divide by zero!";
        if(value > 5) throw 99;
        if(value == 1) throw std::runtime_error("Should be greater than 1 !!!");

        int result = max / value;
        qInfo() << "Result = " << result;


    } catch(std::exception const& e) {
        qWarning() << "We caught the STD way: " << e.what();
        return false;
    } catch(int e) {
        qWarning() << "We caught a number" << e;
        return false;
    } catch (char* e){
        qWarning() << "We caught a pointer to char* " << e;
        return false;
    } catch (...) {
        //Catch all = BAD BAD BAD
        qWarning() << "We know something went wrong, but we dont know what.";
       return false;
    }

    // NO Finally!!!!

    return true;
}

template<typename type>
void print(type value)
{
    qInfo() << "Inside Template function" << value;
}

// template class type T and class type F
// return class type T, parameters are class
// Type T and class Type F
template<class T, class F>
T add(T value1, F value2)
{
    return (value1 + value2);
}

// function test returns void; paramter object pointer of 
// class type QObject; obj 0x1234 (l-value); *obj points 
// to tempory conent (r-value); &obj is 0xABCD
void test(QObject * obj)
{
    qInfo() << obj;
}

void test_QVariant(QVariant value)
{
    qInfo() << value;
    int test = 0;
    bool ok = false;

    test = value.toInt(&ok);

    if (ok)  
        qInfo() << "Int: " << test;
    else
        qInfo() << "Not Int: ";
}

/** Self documenting code using Doxygen.
 * @brief main The starting point
 * @param argc The argument count
 * @param argv The argument
 * @return int The exist value of the application
 */
int main (int argc, char * argv[])
{
    /*
     Print information onto the screen.
     */
    qInfo() << "INFO: Hello World ...";
    qInfo("INFO: Second Hello World ...");

    QCoreApplication a(argc, argv);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    // QString path = QProcessEnvironment::systemEnvironment().value("pwd");
    QString path = env.value("Path");
    qDebug() << "Path Values: " 
             << path;

    {
        // Pointer object of class type scl::CRacecar; player1 == 0x1234 (l-value)
        // *player1 is the temp content (r-value); &player1 is == 0xABCD

        // variable a is (l-value) with content of QCoreApplication class type (r-value)
        // &a is address 0xABCD
        scl::CRacecar * ptr_racecar = new scl::CRacecar(&a);
        test_drive(ptr_racecar);  //it is implecitly converted.
 
        // dynamic cast is a template function holding a pointer object
        // of class type CCar; the parameter passed is a pointer object 
        // of class type CRacecar.  This is an example of casting up from
        // a drived class to parent class.  This is an explicit conversion.
        // dynamic casting will check and provide a nullptr if the conversion
        // failed.
        scl::CCar * ptr_car = dynamic_cast<scl::CCar*>(ptr_racecar);
        if (ptr_car != nullptr)
            test_drive(ptr_car);

        // delete ptr_car;
        ptr_car = nullptr;

        // pointer object of class type scl::CFeline;  ptr_feline has an address
        // 0x1234 (l-value); *ptr_feline is pointing to a temp content of 'CFeline'
        // class type; &ptr_feline is addres of pointer object 0xABCD;
        // dynamic_cast is a function operating on a pointer object of 'CRacecar'
        // class type with a parameter of object class type CRace car.
        // will not work, since CFeline and CRacecar are un-related.
        // scl::CFeline * ptr_feline = dynamic_cast<scl::CRacecar*>(ptr_racecar);
        // if (ptr_racecar != nullptr)
        // {
        //     test_drive(ptr_feline);
        // }

        delete ptr_racecar;
        ptr_racecar = nullptr;
        qInfo() << "Good Bye ...";
    }

    {
        // Static Cast;
        // The static cast operator perform a non-polymorphic cast.
        // For example, it can be used to cast a base clas pointer to derived class pointer
        qInfo() << "\n\n";

        scl::CRacecar * ptr_racecar = new scl::CRacecar(&a);
        test_drive(ptr_racecar);

        scl::CCar * ptr_car = dynamic_cast<scl::CCar*>(ptr_racecar);
        if (ptr_car != nullptr)
            test_drive(ptr_car);

        // up casting.
        // Pointer object of class type 'CRacercar'; ptr_speedster = 0x1234
        // l-value; *ptr_speedster points to object of class type 'CRacer' (r-value)
        // *ptr_speedster is address of pointer object 0xABCD;
        // static_cast is an operator; it operates of pointer object of class type 
        // 'CCar' whose parameter is an ojbect of 'CCar' class type.
        scl::CRacecar * ptr_speedster = static_cast<scl::CRacecar*>(ptr_car);
        if (ptr_speedster != nullptr)
            test_raceCar(ptr_speedster);

        qInfo() << "Good Bye ...";
    }

    {
        // reinterpret cast
        // Unlike static_cast but like const_cast, the reinterpret_cast expression does not compile to ny CPU
        // instructions.  it is purley a compile time directive which instructs the compiler to treat
        // expression as if it had a new_type
        // 
        qInfo() << "\n\n";
        scl::CRacecar * player1 = new scl::CRacecar(&a);

        scl::CCar * obj = dynamic_cast<scl::CCar *>(player1);
        if (obj)
            test_drive(obj);

        scl::CRacecar * speedster = static_cast<scl::CRacecar *>(obj);
        if (speedster)
            test_raceCar(speedster);

        int * pointer = reinterpret_cast<int*>(speedster);
        qInfo() << "Pointer content: " << pointer;

        scl::CRacecar * mycar = reinterpret_cast<scl::CRacecar*>(pointer);
        qInfo() << "Pointer content: " << mycar;
        qInfo() << "Car color: " << mycar->color_;

        qInfo() << "Good Bye ...";
    }

    {
        qInfo() << "\n\n";
        
        //Derived to base ...
        scl::CRacecar * player1 = new scl::CRacecar(&a);
        scl::CCar * mycar = qobject_cast<scl::CCar*>(player1);
        mycar->drive();

        //Base to derived ...
        scl::CRacecar * fastcar = qobject_cast<scl::CRacecar *>(mycar);
        fastcar->go_fast();

        // will not use non-qt objects
        // scl::CDog * fido = qobject_cast<scl::CDog *>(fastcar);
        // Q_UNUSED(fido);

        qInfo() << "Good Bye ...";
    }

    {
        qInfo() << "\n\n";
        int max = 44;
        do 
        {
            // Do something here
        } while(doDivision(max));
               

        qInfo() << "Good Bye ...";
    }

    {
        qInfo() << "\n\n";
        int max = 44;
        
        // function template holds data type integer
        // parameter 1 is passed on.
        print<int>(1);
        print<double>(1.1);
        print<QString>("Hello world");

        // call template function add holding 
        // class type int and class type double
        // parameters are 1 and 4.6
        qInfo() <<  add<int,double>(1, 4.6);

        // scl's template class scl::Ctest holds integer
        // object name is int_calc
        scl::CTest<int> int_calc;
        
        // object of scl's class type CTest has a 
        // function 'add'.  Paramters are two integers.
        qInfo() << int_calc.add(1, 4);

        scl::CTest<double> double_calc;
        qInfo() << double_calc.add(1.1, 4.4);

        // 'scl' temlate class type 'CTest' holds QStrings; parameter name string_calc;
        // template class type 'CTest' has a function with types QString.
        scl::CTest<QString> string_calc;
        qInfo() << string_calc.add("hello", "world");

        qInfo() << "Good Bye ...";
    }

    {
        qInfo() << "\n\n";
        scl::CCar my_car;
        scl::CAnimal my_animal;

        my_car.setObjectName("Fluffy");
        my_animal.setObjectName("Daoggy");

        test(&my_car);
        test(&my_animal);

        qInfo() << "Good Bye ...";
    }

    {
        // typedef - remake a type
        qInfo() << "\n\n";
        qint8 value8 = 0;
        qint16 value16 = 0;
        qint32 value32 = 0;
        qint64 value64 = 0;
        qintptr ptr_value = 0;

        qInfo() << "Value " << sizeof(value8);
        qInfo() << "Value " << sizeof(value16);
        qInfo() << "Value " << sizeof(value32);
        qInfo() << "Value " << sizeof(value64);

        qInfo() << "Good Bye ...";
    }

    {
        // typedef - remake a type
        qInfo() << "\n\n";
        QString name = "Yonder the Wonder";

        qInfo() << name;
        qInfo() << name.mid(1,3);
        qInfo() << name.insert(0,"Mr. ");
        qInfo() << name.split(" ");

        int index = name.indexOf(".");
        qInfo() << name.remove(0,index + 1).trimmed();

        QString title = "teacher";
        QString full = name.trimmed() + " " + title;
        qInfo() << full;

        qInfo() << full.toLatin1();
        qInfo() << "Good Bye ...";
    }

    {
        QDate today = QDate::currentDate();
        qInfo() << today;
        qInfo() << today.addDays(1);
        qInfo() << today.addYears(20);
        qInfo() << today.toString(Qt::DateFormat::SystemLocaleDate);
        qInfo() << today.toString(Qt::DateFormat::TextDate);
        QTime now = QTime::currentTime();
        qInfo() << now;
        qInfo() << now.toString(Qt::DateFormat::DefaultLocaleLongDate);
        qInfo() << now.toString(Qt::DateFormat::DefaultLocaleShortDate);

        QDateTime current = QDateTime::currentDateTime();
        qInfo() << "current: " << current;

        QDateTime expire = current.addDays(45);
        qInfo() << "expire: " << expire;

        if(current > expire) {
            qInfo() << "Expired!";
        } else {
            qInfo() << "Not expired";
        }
    }

    {
        // typedef - remake a type
        qInfo() << "\n\n";

        QString greeting = "Hello World!";
        QByteArray buffer(greeting.toLatin1());
        buffer.append(":");

        qInfo() << "Buffer: " << buffer;

        qInfo() << buffer.rightJustified(20,'.');
        qInfo() << buffer.at(buffer.length()-1);
        QString modified(buffer);

        qInfo() << modified;

        qInfo() << "Good Bye ...";
    }

    {
        // typedef - remake a type
        qInfo() << "\n\n";
        QVariant value = 1;
        QVariant value2 = "Hello world";
        qInfo() << value;
        qInfo() << value2;

        test_QVariant(value);
        test_QVariant(value2);

        qInfo() << "Good Bye ...";
    }

    {
        // typedef - remake a type
        qInfo() << "\n\n";
        QString data = "hello world how are you";
        QStringList lst = data.split(" ");

        qInfo() << lst;
        
        lst.sort();
        foreach (QString item, lst)
            qInfo() << item;

        lst.sort(Qt::CaseInsensitive);
        foreach (QString item, lst)
            qInfo() << item;

        QString my_var = "Hello";
        if(lst.contains(my_var))
        {
            int index = lst.indexOf(my_var);
            qInfo() << lst.value(index);
        }

        qInfo() << "Good Bye ...";
    }

    {
        // typedef - remake a type
        qInfo() << "\n\n";
        QString data = "Hello world";

        // QList is a Qt template class; It holds, contains,
        // QString; the object variable is lst;
        QList<QString> lst = data.split(" ");
        qInfo() << data;
        qInfo() << lst;

        lst.insert(3,"ZZZ");
        foreach(QString item, lst)
        {
            qInfo() << item;
        }

        QList<int> ages({44, 56, 21, 13});
        foreach(int age, ages)
            qInfo() << age;

        qInfo() << "Good Bye ...";
    }

    {
        // typedef - remake a type
        qInfo() << "\n\n";
        QString data = "Hello world";

        QVector<int> ages({44, 56, 21, 13});
        
        qInfo() << ages;
        foreach(int age, ages)
        {
            qInfo() << age;
        }

        qInfo() << "Good Bye ...";
    }

    return a.exec();
}