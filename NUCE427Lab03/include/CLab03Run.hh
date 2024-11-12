#ifndef RUN_HH
#define RUN_HH

#include "G4Run.hh"
#include "Hist.hh"

namespace NUCE427LAB03
{

class CLab03DetectorConstructor;
class CLab03PrimaryGeneratorAction;

class CLab03Run : public G4Run
{

public:
   CLab03Run( CLab03DetectorConstructor* detector_construction,
              CLab03PrimaryGeneratorAction* primary_generator);
   virtual ~CLab03Run();

   virtual void Merge(const G4Run * current_run);
   void EndOfRunSummary();
   void fill_energyDepositInTarget(G4double energydeposit_perEvent) 
   {
      fenergy_depositHistogram->Fill( energydeposit_perEvent );
   }
   
private:
   CLab03DetectorConstructor*   fdetector_constructor;
   CLab03PrimaryGeneratorAction* fprimary_generator;

   // energy deposit in the target
   Hist*   fenergy_depositHistogram;
};

}

#endif