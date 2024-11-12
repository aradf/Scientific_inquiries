#include "CLab03ActionInitialization.hh"
#include "CLab03DetectorConstruction.hh"
#include "CLab03PrimaryGeneratorAction.hh"
#include "CLab03SteppingAction.hh"
#include "CLab03EventAction.hh"
#include "CLab03RunAction.hh"

namespace NUCE427LAB03
{

CLab03ActionInitialization::CLab03ActionInitialization(CLab03DetectorConstructor* detector_constructor) : G4VUserActionInitialization()
{
   fdetector_constructor = detector_constructor;
}

CLab03ActionInitialization::~CLab03ActionInitialization()
{

    
}

void CLab03ActionInitialization::BuildForMaster() const
{
   CLab03RunAction* run_action = new CLab03RunAction( fdetector_constructor,
                                                      nullptr );

   SetUserAction ( run_action );
}

void CLab03ActionInitialization::Build() const
{
   CLab03PrimaryGeneratorAction* primary_generator = new CLab03PrimaryGeneratorAction( fdetector_constructor );
   SetUserAction ( primary_generator );

   CLab03EventAction* event_action = new CLab03EventAction();
   SetUserAction( event_action );

   CLab03SteppingAction* stepping_action = new CLab03SteppingAction( fdetector_constructor, 
                                                                     event_action );
   SetUserAction ( stepping_action );

   CLab03RunAction* run_action = new CLab03RunAction( fdetector_constructor,
                                                      primary_generator );

   SetUserAction( run_action );
}



}