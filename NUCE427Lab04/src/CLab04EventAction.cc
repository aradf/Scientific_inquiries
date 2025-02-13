#include <CLab04EventAction.hh>

#include "G4Event.hh"
#include "G4AnalysisManager.hh"


namespace NUCE427LAB04
{

CLab04EventAction::CLab04EventAction(CLab04RunAction* run_action)
{
    fenergy_deposit = 0.0;
}

CLab04EventAction::~CLab04EventAction()
{
    
}

void CLab04EventAction::BeginOfEventAction(const G4Event* some_event)
{
    fenergy_deposit = 0.0;

}
    
void CLab04EventAction::EndOfEventAction(const G4Event* some_event )
{
    G4cout << "Energy Deposition:: "
           << fenergy_deposit
           << G4endl;
    
    G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();
    analysis_manager->FillNtupleDColumn(1, 0, fenergy_deposit);
    analysis_manager->AddNtupleRow(1);
}



}



