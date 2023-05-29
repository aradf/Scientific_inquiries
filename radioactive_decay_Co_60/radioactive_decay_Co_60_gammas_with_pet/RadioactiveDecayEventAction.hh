#ifndef RADIOACTIVE_DECAY_EVENTACTION_HH
#define RADIOACTIVE_DECAY_EVENTACTION_HH

#include "G4UserEventAction.hh"
#include "G4Event.hh"
#include "G4AnalysisManager.hh"
#include "RadioactiveRunAction.hh"

class CRadioactiveDecayEventAction : public G4UserEventAction
{

public:
   CRadioactiveDecayEventAction(CRadioactiveDecayRunAction *);
   ~CRadioactiveDecayEventAction();

   virtual void BeginOfEventAction(const G4Event *);
   virtual void EndOfEventAction(const G4Event *);

   void add_energy_deposite(G4double energy_deposite) {fEdep += energy_deposite;}

private:
   G4double fEdep;
};

#endif