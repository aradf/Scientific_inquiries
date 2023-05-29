#ifndef RADIOACTIVEDECAY_TRACKINGACTION_HH
#define RADIOACTIVEDECAY_TRACKINGACTION_HH 

#include "G4UserTrackingAction.hh"
#include "globals.hh"

class CRadioactiveDecayEventAction;

class CRadioactiveDecayTrackingAction : public G4UserTrackingAction {

  public:  
    CRadioactiveDecayTrackingAction(CRadioactiveDecayEventAction*);
   ~CRadioactiveDecayTrackingAction();
   
    virtual void PreUserTrackingAction(const G4Track*);
    virtual void PostUserTrackingAction(const G4Track*);
    
  private:
    CRadioactiveDecayEventAction*        fevent;
    
    G4double fcharge, fmass;        
    G4bool   fFullChain;

};



#endif