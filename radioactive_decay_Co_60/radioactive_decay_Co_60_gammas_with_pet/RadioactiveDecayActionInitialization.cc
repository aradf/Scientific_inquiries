#include "RadioactiveDecayActionInitialization.hh"
#include "RadioactiveDecayTrackingAction.hh"

CRadioactiveDecayActionInitialization::CRadioactiveDecayActionInitialization()
{

}

CRadioactiveDecayActionInitialization::~CRadioactiveDecayActionInitialization()
{

}

/**
 * The build virtual function runs the particle gun and computes the stepping.
 * The build vritual function is invoked by G4RunManager for sequential execution.
 */
void CRadioactiveDecayActionInitialization::Build() const
{

   /**
    * User action class primary generator, run action, event action, steping action is defined using 
    * the protected method SetUserAction.
    */

   CRadioactiveDecayPrimaryGenerator * primary_generator = new CRadioactiveDecayPrimaryGenerator();
   SetUserAction(primary_generator);

   CRadioactiveDecayRunAction * run_action = new CRadioactiveDecayRunAction();
   SetUserAction(run_action);

   CRadioactiveDecayEventAction * event_action = new CRadioactiveDecayEventAction(run_action);
   SetUserAction(event_action);

   CRadioactiveDecaySteppingAction * stepping_action = new CRadioactiveDecaySteppingAction(event_action);
   SetUserAction(stepping_action);

   CRadioactiveDecayTrackingAction * tracking_action = new CRadioactiveDecayTrackingAction(event_action);
   SetUserAction(tracking_action);

}

