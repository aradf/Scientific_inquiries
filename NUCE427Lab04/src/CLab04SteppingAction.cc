#include <CLab04SteppingAction.hh>
#include <CLab04DetectorConstruction.hh>

#include <G4Step.hh>
#include <G4RunManager.hh>

namespace NUCE427LAB04
{

CLab04SteppingAction::CLab04SteppingAction(CLab04EventAction* event_action)
{
    fevent_action = event_action;
}

CLab04SteppingAction::~CLab04SteppingAction()
{


}

void CLab04SteppingAction::UserSteppingAction( const G4Step* current_step)
{
    G4LogicalVolume* current_logicalVolume = current_step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
    
    const CLab04DetectorConstructor* detector_construction = static_cast< const CLab04DetectorConstructor* > (G4RunManager::GetRunManager()->GetUserDetectorConstruction());

    G4LogicalVolume* scoring_logicalVolume = detector_construction->getscoring_logicalVolume();
    if( current_logicalVolume == scoring_logicalVolume )
    {
        G4double energy_deposit = current_step->GetTotalEnergyDeposit();
        fevent_action->add_energyDeposit( energy_deposit );
    }
}


}