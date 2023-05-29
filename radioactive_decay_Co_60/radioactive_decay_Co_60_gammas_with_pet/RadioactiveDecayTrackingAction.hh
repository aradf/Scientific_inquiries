#ifndef RADIOACTIVE_DECAY_TRACKING_ACTION_HH
#define RADIOACTIVE_DECAY_TRACKING_ACTION_HH

#include "G4UserTrackingAction.hh"
#include "RadioactiveDecayEventAction.hh"

class CRadioactiveDecayTrackingAction : public G4UserTrackingAction 
{

public:  
  CRadioactiveDecayTrackingAction(CRadioactiveDecayEventAction * ) ;
  ~CRadioactiveDecayTrackingAction();
   
virtual void PreUserTrackingAction(const G4Track*);
virtual void PostUserTrackingAction(const G4Track*);
    
private:
    // EventAction*        fEvent;
    CRadioactiveDecayEventAction * fevent_action;
    // G4double fCharge, fMass;        
    G4double fcharge, fmass;
};


#endif
