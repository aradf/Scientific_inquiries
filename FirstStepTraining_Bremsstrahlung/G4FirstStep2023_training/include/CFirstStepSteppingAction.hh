#ifndef STEPPING_ACTION_HH
#define STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"

namespace FS
{

class CFirstStepDetectorConstruction;
class CFirstStepEventAction;

class CFirstStepSteppingAction : public G4UserSteppingAction
{

public:
   CFirstStepSteppingAction(CFirstStepDetectorConstruction * detector_construction, 
                            CFirstStepEventAction * event_action);
   virtual ~CFirstStepSteppingAction();

   virtual void UserSteppingAction(const G4Step *);

private:

    CFirstStepDetectorConstruction * fdetector_constructor;
    CFirstStepEventAction * fevent_action;
};

}

#endif
