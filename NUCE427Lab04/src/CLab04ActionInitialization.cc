#include <CLab04ActionInitialization.hh>
#include <CLab04PrimaryGeneratorAction.hh>
#include <CLab04RunAction.hh>
#include <CLab04EventAction.hh>
#include <CLab04SteppingAction.hh>

#include "globals.hh"

namespace NUCE427LAB04
{

CLab04ActionInitialization::CLab04ActionInitialization() : G4VUserActionInitialization()
{
    G4cout << "INFO: CLab04ActionInitialization Constructor ..."
           << G4endl;


}

CLab04ActionInitialization::~CLab04ActionInitialization()
{


}

void CLab04ActionInitialization::Build() const
{
   CLab04PrimaryGeneratorAction* primary_generator = new CLab04PrimaryGeneratorAction();
   SetUserAction ( primary_generator );

   CLab04RunAction* run_action = new CLab04RunAction();
   SetUserAction ( run_action );

   CLab04EventAction* event_action = new CLab04EventAction( run_action );
   SetUserAction ( event_action );

   CLab04SteppingAction* stepping_action = new CLab04SteppingAction( event_action );
   SetUserAction ( stepping_action );

}
    
void CLab04ActionInitialization::BuildForMaster() const
{

}

}


