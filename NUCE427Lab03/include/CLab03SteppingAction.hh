#ifndef STEPPING_ACTION_HH
#define STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"

namespace NUCE427LAB03
{

class CLab03DetectorConstructor;
class CLab03EventAction;

class CLab03SteppingAction : public G4UserSteppingAction
{
public:
   CLab03SteppingAction( CLab03DetectorConstructor* detector_construction, CLab03EventAction* event_action );
   virtual ~CLab03SteppingAction();
   virtual void UserSteppingAction( const G4Step* );
private:
   CLab03DetectorConstructor* fdetector_constructor;
   CLab03EventAction* fevent_action; 
};

}

#endif
