#ifndef SHAREDDOG_HH                    // The compiler understand to define if it is not defined.
#define SHAREDDOG_HH                    // Pre process directive.

#include <QObject>                   // Gives access to signals and slots.
#include <QDebug>                    // gives access to qInfo()

#include <memory>                   // Gives access to std::shared_ptr
#include <string>                   // Gives access to std::string

// Scientific Computational Library
namespace scl
{
class CSharedDog
{
    std::shared_ptr<CSharedDog> shared_ptrDog;
    std::weak_ptr<CSharedDog> weak_ptrDog;
    std::string name_;
public:
    CSharedDog();
    CSharedDog(std::string name);
     ~CSharedDog();
    void bark();
    void make_friend(std::shared_ptr<CSharedDog> ptr_sharedDog);
    void show_friend();
};

} // end of Scientific Computational Library.

#endif 