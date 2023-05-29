#ifndef RADIOACTIVE_DECAY_RUN_ACTION_HH
#define RADIOACTIVE_DECAY_RUN_ACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "G4AnalysisManager.hh"

class CRadioactiveDecayRunAction : public G4UserRunAction
{
public:
   CRadioactiveDecayRunAction();
   ~CRadioactiveDecayRunAction();

   virtual void BeginOfRunAction(const G4Run *);
   virtual void EndOfRunAction(const G4Run *);
};


#endif