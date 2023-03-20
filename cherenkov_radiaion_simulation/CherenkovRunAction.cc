#include "CherenkovRunAction.hh"

CCherenkovRunAction::CCherenkovRunAction()
{
    G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();
    manager_analysis->CreateNtuple("Photons", "Photons");
    manager_analysis->CreateNtupleIColumn("fEvent");
    manager_analysis->CreateNtupleDColumn("fX");
    manager_analysis->CreateNtupleDColumn("fY");
    manager_analysis->CreateNtupleDColumn("fZ");
    manager_analysis->CreateNtupleDColumn("fWaveLen");
    manager_analysis->FinishNtuple(0);

    manager_analysis->CreateNtuple("Hits", "Hits");
    manager_analysis->CreateNtupleIColumn("fEvent");
    manager_analysis->CreateNtupleDColumn("fX");
    manager_analysis->CreateNtupleDColumn("fY");
    manager_analysis->CreateNtupleDColumn("fZ");
    manager_analysis->FinishNtuple(1);

    manager_analysis->CreateNtuple("Scoring", "Scoring");
    manager_analysis->CreateNtupleDColumn("fEdep");
    manager_analysis->FinishNtuple(2);

}

CCherenkovRunAction::~CCherenkovRunAction()
{


}

/**
 * This method is invoked before converting the primary particles to a G4Track object.
 * Initialize and/or open output file for a particular event (data storage).
 */

void CCherenkovRunAction::BeginOfRunAction(const G4Run * run_action)
{
   /**
    * What information we want to create, must re-start for every run.
    * Created new for every run.  the run that include the events.
    *  
    */
    G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();
    
    G4int run_number = run_action->GetRunID();
    std::stringstream str_runID;
    str_runID << run_number;

    manager_analysis->OpenFile("cherenkov_output" + str_runID.str() + ".root");
}

/**
 * This method is invoked at the very end of event processing.  it is simply used for simple analysis.
 */
void CCherenkovRunAction::EndOfRunAction(const G4Run * run_action)
{
    G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();
    manager_analysis->Write();
    manager_analysis->CloseFile();

}
