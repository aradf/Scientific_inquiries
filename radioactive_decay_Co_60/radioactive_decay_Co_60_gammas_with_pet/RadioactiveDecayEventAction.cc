#include "RadioactiveDecayEventAction.hh"

CRadioactiveDecayEventAction::CRadioactiveDecayEventAction(CRadioactiveDecayRunAction *)
{
   fEdep = 0.0;
}

CRadioactiveDecayEventAction::~CRadioactiveDecayEventAction()
{

}

void CRadioactiveDecayEventAction::BeginOfEventAction(const G4Event *)
{
   fEdep = 0.0;
}

void CRadioactiveDecayEventAction::EndOfEventAction(const G4Event *)
{
   G4cout << "Energy Deposition: " << fEdep << " MeV" << G4endl;

   G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();
   manager_analysis ->FillNtupleDColumn(2, 0, fEdep);
   manager_analysis->AddNtupleRow(2);
}
