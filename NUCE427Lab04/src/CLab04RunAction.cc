#include <CLab04RunAction.hh>

#include <G4Run.hh>
#include <G4AnalysisManager.hh>
// #include <G4root.hh>

namespace NUCE427LAB04
{

CLab04RunAction::CLab04RunAction()
{
    G4cout << "INFO: CLab04RunAction Constructor ..."
           << G4endl;

   G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();
   analysis_manager->CreateNtuple("Photons", "Photons");
   analysis_manager->CreateNtupleIColumn("fEvent");
   analysis_manager->CreateNtupleDColumn("fX");
   analysis_manager->CreateNtupleDColumn("fY");
   analysis_manager->CreateNtupleDColumn("fZ");
   analysis_manager->CreateNtupleDColumn("fWlen");
   analysis_manager->FinishNtuple(0);

   analysis_manager->CreateNtuple("Scoring","Scoring");
   analysis_manager->CreateNtupleDColumn("fEnergyDepsit");
   analysis_manager->FinishNtuple(1);

}

CLab04RunAction::~CLab04RunAction()
{


}

void CLab04RunAction::BeginOfRunAction(const G4Run* current_run)
{
   G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();
   
   G4int run_id = current_run->GetRunID();
   std::stringstream string_runID;
   string_runID << run_id;
   G4String file_name = "output" + string_runID.str() + ".root";

   // analysis_manager->OpenFile("output.root");
   analysis_manager->OpenFile(file_name);

}


void CLab04RunAction::EndOfRunAction(const G4Run * current_run)
{
   G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();    
   analysis_manager->Write();
   analysis_manager->CloseFile();

}

G4Run* CLab04RunAction::GenerateRun()
{
    G4Run* current_run = nullptr;

    return current_run;
}



}
