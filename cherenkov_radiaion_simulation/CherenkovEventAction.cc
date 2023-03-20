#include "CherenkovEventAction.hh"

CCherenkovEventAction::CCherenkovEventAction(CCherenkovRunAction *)
{
   fEdep = 0;
}

CCherenkovEventAction::~CCherenkovEventAction()
{

}

void CCherenkovEventAction::BeginOfEventAction(const G4Event *)
{
   fEdep = 0;
}

void CCherenkovEventAction::EndOfEventAction(const G4Event *)
{

   G4cout << "Energy Deposition: " << fEdep << " MeV" << G4endl;

   G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();
   manager_analysis ->FillNtupleDColumn(2, 0, fEdep);
   manager_analysis->AddNtupleRow(2);
}

