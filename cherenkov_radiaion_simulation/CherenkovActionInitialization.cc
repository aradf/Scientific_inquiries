#include "CherenkovActionInitialization.hh"

CCherenkovActionInitialization::CCherenkovActionInitialization()
{

}

CCherenkovActionInitialization::~CCherenkovActionInitialization()
{

}

/**
 * The build virtual function runs the particle gun and computes the stepping.
 * The build vritual function is invoked by G4RunManager for sequential execution.
 */
void CCherenkovActionInitialization::Build() const
{

   /**
    * User action class primary generator, run action, event action, steping action is defined using 
    * the protected method SetUserAction.
    */
   CCherenkovPrimaryGenerator * primary_generator = new CCherenkovPrimaryGenerator();
   SetUserAction(primary_generator);

   CCherenkovRunAction * run_action  = new CCherenkovRunAction();
   SetUserAction(run_action);

   CCherenkovEventAction * event_action = new CCherenkovEventAction(run_action);
   SetUserAction(event_action);

   CCherenkovSteppingAction * stepping_action = new CCherenkovSteppingAction(event_action);
   SetUserAction(stepping_action);
}

