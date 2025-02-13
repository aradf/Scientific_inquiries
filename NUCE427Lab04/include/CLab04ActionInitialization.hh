#ifndef ACTION_INITIALIZATION_HH
#define ACTION_INITIALIZATION_HH

#include <G4VUserActionInitialization.hh>

namespace NUCE427LAB04
{

class CLab04ActionInitialization : public G4VUserActionInitialization
{
public:
    CLab04ActionInitialization();
    virtual ~CLab04ActionInitialization();
    virtual void Build() const;
    virtual void BuildForMaster() const;

};


}



#endif