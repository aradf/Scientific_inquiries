#ifndef RADIOACTIVE_DECAY_RUN_ACTION_HH
#define RADIOACTIVE_DECAY_RUN_ACTION_HH

#include "G4UserRunAction.hh"
#include "G4AnalysisManager.hh"
#include "G4Run.hh"
#include "RadioactiveDecayPrimaryGenerator.hh"
#include "RadioactiveDecayHistoManager.hh"
#include "RadioactiveDecayRun.hh"

class CRadioactiveDecayRunAction : public G4UserRunAction
{
public:
   CRadioactiveDecayRunAction(CRadioactiveDecayPrimaryGenerator * );
   ~CRadioactiveDecayRunAction();

   virtual G4Run* GenerateRun();   
   virtual void BeginOfRunAction(const G4Run *);
   virtual void EndOfRunAction(const G4Run *);

private:
   CRadioactiveDecayPrimaryGenerator * fprimary;
   CRadioactiveDecayRun *              fradioactivedecay_run;
   CRadioactiveDecayHistoManager *     fradioactivedecay_histogrammanager;    
};


#endif