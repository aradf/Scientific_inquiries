
#include "CLab03RunAction.hh"
#include "CLab03DetectorConstruction.hh"
#include "CLab03PrimaryGeneratorAction.hh"
#include "CLab03Run.hh"

#include "G4Run.hh"

namespace NUCE427LAB03
{

CLab03RunAction::CLab03RunAction(CLab03DetectorConstructor* detector_constructor,
                                 CLab03PrimaryGeneratorAction* primary_generator) : G4UserRunAction()
{
   fdetector_constructor = detector_constructor;
   fprimary_generator = primary_generator;
   frun = nullptr;
}                   

CLab03RunAction::~CLab03RunAction()
{

}

G4Run* CLab03RunAction::GenerateRun()
{
   frun = new CLab03Run( fdetector_constructor, 
                         fprimary_generator );
   return frun;
}


void CLab03RunAction::BeginRunAction(const G4Run* current_run)
{
   if ( fprimary_generator != nullptr )
   {
      fprimary_generator->update_position();
   }
}

void CLab03RunAction::EndOfRunAction(const G4Run* current_run)
{
   if ( IsMaster() )
   {
      frun->EndOfRunSummary();
   }
}

}
