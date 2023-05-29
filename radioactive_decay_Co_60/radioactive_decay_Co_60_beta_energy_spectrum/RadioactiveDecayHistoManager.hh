#ifndef RADIOACTIVE_DECAY_HISTO_MANAGER_HH
#define RADIOACTIVE_DECAY_HISTO_MANAGER_HH

#include "globals.hh"
#include "G4AnalysisManager.hh"

class CRadioactiveDecayHistoManager
{
public:
   CRadioactiveDecayHistoManager();
  ~CRadioactiveDecayHistoManager();

private:
    void configure();
    G4String ffile_name;
};

#endif

