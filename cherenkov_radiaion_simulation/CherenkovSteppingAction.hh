#ifndef CHERENKOV_STEPPING_HH
#define CHERENKOV_STEPPING_HH

#include "G4UserSteppingAction.hh"
#include "G4Step.hh"

#include "CherenkovDetectorConstruction.hh"
#include "CherenkovEventAction.hh"

/**
 * This class represents actions taken place by the user at each end of stepping.
 * The Geant4 developer can intervene in the tracking object using the G4UserSteppingAction
 * objects, members, methods.
 */
class CCherenkovSteppingAction : public G4UserSteppingAction
{
public:
   CCherenkovSteppingAction(CCherenkovEventAction * event_action);
   ~CCherenkovSteppingAction();

   virtual void UserSteppingAction(const G4Step *);

private:
   CCherenkovEventAction * fevent_action;
};

#endif