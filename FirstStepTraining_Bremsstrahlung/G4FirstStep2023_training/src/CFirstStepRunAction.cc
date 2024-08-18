#include "CFirstStepDetectorConstruction.hh"
#include "CFirstStepRunAction.hh"
#include "CFirstStepPrimaryGeneratorAction.hh"
#include "CFirstStepRun.hh"

#include "G4Run.hh"

namespace FS
{

CFirstStepRunAction::CFirstStepRunAction(CFirstStepDetectorConstruction * detector_construction,
                                         CFirstStepPrimaryGeneratorAction * primary_generator)                  
                                         : G4UserRunAction()
{
   fdetector_constructor = detector_construction;
   fprimary_generator = primary_generator;
   frun = nullptr;
}

G4Run * CFirstStepRunAction::GenerateRun()
{
   frun = new CFirstStepRun( fdetector_constructor , 
                             fprimary_generator );

   return frun;
}

CFirstStepRunAction::~CFirstStepRunAction()
{
   // if (frun != nullptr)
   // {
   //    delete frun;
   //    frun = nullptr;
   // }
}

void CFirstStepRunAction::BeginRunAction(const G4Run * /* current_run */)
{
   if ( fprimary_generator != nullptr )
   {
      fprimary_generator->update_position();
   }
      
}

void CFirstStepRunAction::EndOfRunAction(const G4Run * /* current_run */)
{
   if ( IsMaster() )
   {
      frun->EndOfRunSummary();
   }
}

};

