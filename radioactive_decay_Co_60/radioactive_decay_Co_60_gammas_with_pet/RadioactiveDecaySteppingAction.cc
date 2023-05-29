#include "RadioactiveDecaySteppingAction.hh"


CRadioactiveDecaySteppingAction::CRadioactiveDecaySteppingAction(CRadioactiveDecayEventAction * event_action)
{
   fevent_action = event_action;
}

CRadioactiveDecaySteppingAction::~CRadioactiveDecaySteppingAction()
{

}

void CRadioactiveDecaySteppingAction::UserSteppingAction(const G4Step * step)
{
   G4LogicalVolume * volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();

   const CRadioactiveDecayDetectorConstruction * detector_construction = static_cast<const CRadioactiveDecayDetectorConstruction*> (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    
   G4LogicalVolume *fScoringVolume = detector_construction->get_scoring_volume();
    
   if(volume != fScoringVolume)
       return;

   G4double energy_deposite = step->GetTotalEnergyDeposit();
   fevent_action->add_energy_deposite(energy_deposite);
}

