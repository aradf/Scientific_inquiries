#ifndef RUN_ACTION_HH
#define RUN_ACTION_HH

#include "G4UserRunAction.hh"

namespace NUCE427LAB03
{

class CLab03DetectorConstructor;
class CLab03PrimaryGeneratorAction;
class CLab03Run;

class CLab03RunAction : public G4UserRunAction
{
public:    
   CLab03RunAction(CLab03DetectorConstructor* detector_constructor,
                   CLab03PrimaryGeneratorAction* primary_generator);
   virtual ~CLab03RunAction();

   virtual void BeginRunAction(const G4Run * current_run);
   virtual void EndOfRunAction(const G4Run * current_run);

   virtual G4Run* GenerateRun();
private:

   CLab03DetectorConstructor* fdetector_constructor;
   CLab03PrimaryGeneratorAction* fprimary_generator;
   CLab03Run* frun;
};

}

#endif
