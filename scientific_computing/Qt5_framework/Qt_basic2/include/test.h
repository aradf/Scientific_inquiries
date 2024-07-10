#ifndef TEST_HH                    // The compiler understand to define if it is not defined.
#define TEST_HH                    // Pre process directive.

// Scientific Computational Library
namespace scl
{

// template class T
// 
template<typename T>
class CTest 
{
public:
    explicit CTest()
    {
        qInfo() << this << "Constructor: CTest invoked ...";
    }
    virtual ~CTest()
    {
        qInfo() << this << "Destructor: CTest invoked ...";
    }
    T add(T value1, T value2) {return value1 + value2;};
    
};

} // end of Scientific Computational Library.

#endif 