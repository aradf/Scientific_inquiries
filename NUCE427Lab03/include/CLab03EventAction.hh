#ifndef EVENT_ACTION_HH
#define EVENT_ACTION_HH

#include "G4UserEventAction.hh"
#include "globals.hh"

namespace NUCE427LAB03
{

class CLab03EventAction : public G4UserEventAction
{
public:
   CLab03EventAction();
   ~CLab03EventAction();

   virtual void BeginOfEventAction(const G4Event* an_event);
   virtual void EndOfEventAction(const G4Event* an_event);

   void add_energyDepositePerEvent(G4double energy_depositeInStep)
   {
      fenergy_depositePerEvent = fenergy_depositePerEvent + energy_depositeInStep;
   }
private:
   G4double fenergy_depositePerEvent;

};


}

#endif