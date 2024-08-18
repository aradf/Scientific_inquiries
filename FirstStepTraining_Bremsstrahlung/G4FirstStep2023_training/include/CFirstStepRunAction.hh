#ifndef RUN_ACTION_HH
#define RUN_ACTION_HH

#include "G4UserRunAction.hh"

namespace FS
{

class CFirstStepDetectorConstruction;
class CFirstStepPrimaryGeneratorAction;
class CFirstStepRun;

class CFirstStepRunAction : public G4UserRunAction
{

public:
   CFirstStepRunAction(CFirstStepDetectorConstruction * detector_construction, 
                       CFirstStepPrimaryGeneratorAction * primary_generator);
   virtual ~CFirstStepRunAction();
   
   virtual void BeginRunAction(const G4Run * current_run);
   virtual void EndOfRunAction(const G4Run * current_run);

   virtual G4Run * GenerateRun();
private:

   CFirstStepDetectorConstruction * fdetector_constructor;
   CFirstStepPrimaryGeneratorAction * fprimary_generator;
   CFirstStepRun * frun;
};

}

#endif
