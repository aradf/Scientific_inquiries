#ifndef STEPPING_ACTION_HH
#define STEPPING_ACTION_HH

#include <G4UserSteppingAction.hh>

#include <CLab04EventAction.hh>

class G4Step;


namespace NUCE427LAB04
{

class CLab04SteppingAction : public G4UserSteppingAction
{
public:
    CLab04SteppingAction(CLab04EventAction* );
    ~CLab04SteppingAction();

    virtual void UserSteppingAction( const G4Step* );

private:
    CLab04EventAction* fevent_action;
};


}



#endif 