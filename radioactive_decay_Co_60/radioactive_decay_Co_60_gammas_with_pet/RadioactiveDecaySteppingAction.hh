#ifndef RADIOACTIVE_DECAY_STEPPING_ACTION_HH
#define RADIOACTIVE_DECAY_STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"
#include "G4Step.hh"

#include "RadioactiveDecayDetectorConstruncion.hh"
#include "RadioactiveDecayEventAction.hh"


class CRadioactiveDecaySteppingAction : public G4UserSteppingAction
{
public:
   CRadioactiveDecaySteppingAction(CRadioactiveDecayEventAction * event_action);
   ~CRadioactiveDecaySteppingAction();

   virtual void UserSteppingAction(const G4Step *);

private:
   CRadioactiveDecayEventAction * fevent_action;
};

#endif