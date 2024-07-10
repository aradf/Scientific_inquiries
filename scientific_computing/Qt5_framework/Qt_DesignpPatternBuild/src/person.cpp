#include "person.h"

CPerson *CPerson::build(CPerson::persontype type)
{
    // 
    CPerson *ptr_person = new CPerson(nullptr);
    ptr_person->m_type_ = type;

    return ptr_person;
}

CPerson::persontype CPerson::role()
{
    return m_type_;
}

CPerson::CPerson(QObject *parent) : QObject(parent)
{

}
