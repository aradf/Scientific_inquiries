#ifndef RADIOACTIVE_DECAY_TRACKING_MESSENGER_HH
#define RADIOACTIVE_DECAY_TRACKING_MESSENGER_HH

#include "globals.hh"
#include "G4UImessenger.hh"

#include "G4UIcmdWithABool.hh"
#include "G4UIcommand.hh"
#include "G4UIparameter.hh"

#include "RadioactiveDecayTrackingAction.hh"

class CRadioActiveDecayTrackingMessenger: public G4UImessenger
{
  public:
    CRadioActiveDecayTrackingMessenger(CRadioactiveDecayTrackingAction*);
   ~CRadioActiveDecayTrackingMessenger();
    
    virtual void SetNewValue(G4UIcommand*, G4String);
    
  private:
    CRadioactiveDecayTrackingAction*   ftracking_action;    
    G4UIcmdWithABool * ftracking_cmd;

};


#endif