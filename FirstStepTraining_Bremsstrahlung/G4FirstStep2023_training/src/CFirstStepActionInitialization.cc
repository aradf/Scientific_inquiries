#include "CFirstStepActionInitialization.hh"
#include "CFirstStepPrimaryGeneratorAction.hh"
#include "CFirstStepDetectorConstruction.hh"

#include "CFirstStepSteppingAction.hh"
#include "CFirstStepEventAction.hh"
#include "CFirstStepRunAction.hh"

namespace FS
{
    
CFirstStepActionInitialization::CFirstStepActionInitialization(CFirstStepDetectorConstruction *detector_construction) 
                                                              : G4VUserActionInitialization()
{
   fdetector_construction = detector_construction;
}

CFirstStepActionInitialization::~CFirstStepActionInitialization()
{

}

void CFirstStepActionInitialization::BuildForMaster() const
{
   CFirstStepRunAction * run_action = new CFirstStepRunAction ( fdetector_construction, 
                                                                nullptr);
   SetUserAction ( run_action );
}

void CFirstStepActionInitialization::Build() const
{
   CFirstStepPrimaryGeneratorAction * primary_generator = new CFirstStepPrimaryGeneratorAction( fdetector_construction );
   SetUserAction ( primary_generator );

   CFirstStepEventAction * event_action = new CFirstStepEventAction();
   SetUserAction( event_action );

   CFirstStepSteppingAction * stepping_action = new  CFirstStepSteppingAction(fdetector_construction,
                                                                              event_action);
   SetUserAction ( stepping_action );

   CFirstStepRunAction * run_action = new CFirstStepRunAction ( fdetector_construction, 
                                                                primary_generator);
   SetUserAction ( run_action );
}
  

}
