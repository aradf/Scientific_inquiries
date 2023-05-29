#include "RadioactiveRunAction.hh"

CRadioactiveDecayRunAction::CRadioactiveDecayRunAction()
{
   G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();
   manager_analysis->CreateNtuple("Photons", "Photons");
   manager_analysis->CreateNtupleIColumn("fEvent");
   manager_analysis->CreateNtupleDColumn("fX");
   manager_analysis->CreateNtupleDColumn("fY");
   manager_analysis->CreateNtupleDColumn("fZ");
   manager_analysis->CreateNtupleDColumn("fWLen");
   //manager_analysis->CreateNtupleDColumn("fT");
   manager_analysis->FinishNtuple(0);

   manager_analysis->CreateNtuple("Hits", "Hits");
   manager_analysis->CreateNtupleIColumn("fEvent");
   manager_analysis->CreateNtupleDColumn("fX");
   manager_analysis->CreateNtupleDColumn("fY");
   manager_analysis->CreateNtupleDColumn("fZ");
   manager_analysis->FinishNtuple(1);

   manager_analysis->CreateNtuple("Scoring","Scoring");
   manager_analysis->CreateNtupleDColumn("fEdep");
   manager_analysis->FinishNtuple(2);

   manager_analysis->CreateNtuple("BetaEnergySpectrum","BetaEnergySpectrum");
   manager_analysis->CreateNtupleDColumn("fESpec");
   manager_analysis->FinishNtuple(3);

}

CRadioactiveDecayRunAction::~CRadioactiveDecayRunAction()
{

}

void CRadioactiveDecayRunAction::BeginOfRunAction(const G4Run * some_run)
{
   G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();

   G4int runID = some_run->GetRunID();

   std::stringstream str_runID;
   str_runID << runID;

   manager_analysis->OpenFile("output" + str_runID.str() + ".root");
}

void CRadioactiveDecayRunAction::EndOfRunAction(const G4Run *)
{
   G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();

   manager_analysis->Write();
   manager_analysis->CloseFile();
}

