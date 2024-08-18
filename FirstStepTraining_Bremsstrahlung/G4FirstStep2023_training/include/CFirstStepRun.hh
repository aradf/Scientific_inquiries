#ifndef RUN_HH
#define RUN_HH

#include "G4Run.hh"
#include "Hist.hh"

namespace FS
{

class CFirstStepDetectorConstruction;
class CFirstStepPrimaryGeneratorAction;

class CFirstStepRun : public G4Run
{

public:
   CFirstStepRun(CFirstStepDetectorConstruction * detector_construction,
                 CFirstStepPrimaryGeneratorAction * primary_generator);
   virtual ~CFirstStepRun();

   virtual void Merge(const G4Run * current_run);
   void EndOfRunSummary();
   void fill_energyDepositInTarget(G4double energydeposite_perEvent) {
      fenergy_depositHistogram->Fill( energydeposite_perEvent );
   }
   
private:
   CFirstStepDetectorConstruction*   fdetector_construction;
   CFirstStepPrimaryGeneratorAction* fprimary_generator;

   // energy deposit in the target
   Hist*   fenergy_depositHistogram;
};

}

#endif
