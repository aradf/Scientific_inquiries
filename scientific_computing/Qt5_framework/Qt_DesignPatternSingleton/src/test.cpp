#include "test.h"

// Static function member 'createInstance' returns a pointer object of CTest class type.  
CTest * CTest::createInstance()
{
    return new CTest();
}

CTest::CTest(QObject *parent) : QObject(parent)
{

}

// static function member 'instance' returns a pointer object of CTest class type.
CTest * CTest::instance()
{
    // Singlelton is a user-defined templated class defined earlier.  It holds CTest class type;
    return  Singleton<CTest>::instance(CTest::createInstance);
}
