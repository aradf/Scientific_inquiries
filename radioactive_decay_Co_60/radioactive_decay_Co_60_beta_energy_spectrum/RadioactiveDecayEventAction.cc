#include "RadioactiveDecayEventAction.hh"

#include "RadioactiveDecayHistoManager.hh"
#include "G4RunManager.hh"
#include "RadioactiveDecayRun.hh"


CRadioactiveDecayEventAction::CRadioactiveDecayEventAction() :G4UserEventAction(), fdecay_chain(),fevisible_total(0.0) 
{
   // Set default print level 
   G4RunManager::GetRunManager()->SetPrintProgress(10000);
}

CRadioactiveDecayEventAction::~CRadioactiveDecayEventAction()
{

}

/**
 * This method is invoked before converting the primary particles to G4Track objects. A typical use of this 
 * method would be to initialize and/or book histograms for a particular event
 */
void CRadioactiveDecayEventAction::BeginOfEventAction(const G4Event*)
{
   fdecay_chain = G4String(" ");
   fevisible_total = 0.0;
}

/**
 * This method is invoked at the very end of event processing. It is typically used for
 * an analysis of the event. The user can keep the currently processing 
 * event until the end of the current run.
 */
void CRadioactiveDecayEventAction::EndOfEventAction(const G4Event* some_event)
{
   G4int event_number = some_event->GetEventID();
   G4int print_progress = G4RunManager::GetRunManager()->GetPrintProgress(); 
   // printing survey

   if (event_number % print_progress == 0) 
   G4cout << "    End of event. Decay chain:" << fdecay_chain << G4endl << G4endl;

   //total visible energy
   G4AnalysisManager::Instance()->FillH1(9, fevisible_total);
   CRadioactiveDecayRun * run = static_cast<CRadioactiveDecayRun*>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());
   run->evis_event(fevisible_total);
}
