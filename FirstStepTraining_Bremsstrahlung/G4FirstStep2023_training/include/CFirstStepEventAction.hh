#ifndef EVENT_ACTION_HH
#define EVENT_ACTION_HH

#include "G4UserEventAction.hh"
#include "globals.hh"

namespace FS
{

class CFirstStepEventAction : public G4UserEventAction
{

public:
   CFirstStepEventAction();
   ~CFirstStepEventAction();
   
   virtual void BeginOfEventAction(const G4Event * an_event);
   virtual void EndOfEventAction(const G4Event * an_event);

   void add_energyDepsoitePerEvent(G4double energyDeposite_inStep) {
      fenergy_depositePerEvent = fenergy_depositePerEvent + energyDeposite_inStep;
   }
   
private:
   G4double fenergy_depositePerEvent;

};



}

#endif
