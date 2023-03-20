
#include "CherenkovSteppingAction.hh"

CCherenkovSteppingAction::CCherenkovSteppingAction(CCherenkovEventAction * event_action)
{
   fevent_action = event_action;
}

CCherenkovSteppingAction::~CCherenkovSteppingAction()
{

}

/**
 * Actions taken place by the user at each end of stepping.
 * In the UserSteppingAction function some energy deposition and 
 * track length of charged particles are collected for a selected volume.
 */
void CCherenkovSteppingAction::UserSteppingAction(const G4Step * step)
{
   /**
    * return a pointer to the logical volume of the current step.
   */
   G4LogicalVolume * volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();

   /**
    * return a pointer to the detector construnction object
    */
   const CCherenkovDetectorConstruction *detectorConstruction = static_cast<const CCherenkovDetectorConstruction*> (G4RunManager::GetRunManager()->GetUserDetectorConstruction());

   /**
    * return a pointer to the scoring volume that is defined in the detector construction.
    */   
   G4LogicalVolume *fScoringVolume = detectorConstruction->get_scoring_volume();
    
   if(volume != fScoringVolume)
       return;

   /**
    * The total energy deposit is available in the GStep object.  Since all scoring volume is defined and specified,
    * The energy deposite is calculated for the scoring logicla volume only.
    */
   G4double energy_deposite = step->GetTotalEnergyDeposit();
   fevent_action->add_energy_deposite(energy_deposite);
}
