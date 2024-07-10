#include "../include/shared_dog.h"

scl::CSharedDog::CSharedDog()
{
    qInfo() << this << "Constructor: CSharedDog invoked ...";
}

scl::CSharedDog::~CSharedDog()
{
    qInfo() << this << "Destructor: CSharedDog invoked ..." << QString::fromUtf8(name_.c_str());
}

scl::CSharedDog::CSharedDog(std::string name)
{
    this->name_ = name;
    qInfo() << this << "Constructor: CSharedDog invoked ..." << QString::fromUtf8(name_.c_str());
}

void scl::CSharedDog::bark()
{
    qInfo() << this << "CSharedDog Barked ...";
}


void scl::CSharedDog::make_friend(std::shared_ptr<CSharedDog> ptr_sharedDog)
{
    this->shared_ptrDog = ptr_sharedDog;
}

void scl::CSharedDog::show_friend()
{
    if (!this->weak_ptrDog.expired())
    {
        QString temp_str = QString::fromUtf8(weak_ptrDog.lock()->name_.c_str());
        qInfo() << "My friend is: " << temp_str;
    }

        
}