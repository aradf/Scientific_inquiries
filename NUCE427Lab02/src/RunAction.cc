// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
#include "RunAction.hh"
#include "G4Run.hh"
#include "G4AccumulableManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4AnalysisManager.hh"

namespace NUCE427LAB02
{

RunAction::RunAction()
{
  // add new units for dose
  // const G4double milligray = 1.e-3*gray;
  // const G4double microgray = 1.e-6*gray;
  // const G4double nanogray  = 1.e-9*gray;
  // const G4double picogray  = 1.e-12*gray;

  // new G4UnitDefinition("milligray", "milliGy" , "Dose", milligray);
  // new G4UnitDefinition("microgray", "microGy" , "Dose", microgray);
  // new G4UnitDefinition("nanogray" , "nanoGy"  , "Dose", nanogray);
  // new G4UnitDefinition("picogray" , "picoGy"  , "Dose", picogray);

  // Register accumulable to the accumulable manager
  // G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();
  G4cout << "RunAction::RunAction: " << G4endl;
  G4AnalysisManager * analysis_manager = G4AnalysisManager::Instance();

  analysis_manager->CreateNtuple("photons", "photons");
  analysis_manager->CreateNtupleIColumn("fEvent");
  analysis_manager->CreateNtupleDColumn("fX");
  analysis_manager->CreateNtupleDColumn("fY");
  analysis_manager->CreateNtupleDColumn("fZ");
  analysis_manager->CreateNtupleDColumn("fw_len");
  analysis_manager->FinishNtuple(0);

  analysis_manager->CreateNtuple("Hits", "Hits");
  analysis_manager->CreateNtupleIColumn("fEvent");
  analysis_manager->CreateNtupleDColumn("fX");
  analysis_manager->CreateNtupleDColumn("fY");
  analysis_manager->CreateNtupleDColumn("fZ");
  analysis_manager->FinishNtuple(1);

  analysis_manager->CreateNtuple("Scoring","Scoring");
  analysis_manager->CreateNtupleDColumn("fenergy_deposite");
  analysis_manager->FinishNtuple(2);

}

RunAction::~RunAction()
{

}

void RunAction::BeginOfRunAction(const G4Run * simulation_run)
{
  G4cout << "RunAction::BeginOfRunAction: " << G4endl;
  G4AnalysisManager * analysis_manager = G4AnalysisManager::Instance();
  G4int run_number = simulation_run->GetRunID();
  std::stringstream str_runID;
  str_runID << run_number;

  analysis_manager->OpenFile("output_" + str_runID.str() + ".csv");
  //analysis_manager->OpenFile("output_" + str_runID.str() + ".root");

}

void RunAction::EndOfRunAction(const G4Run* run)
{
  G4cout << "RunAction::EndOfRunAction: " << G4endl;
  G4int nofEvents = run->GetNumberOfEvent();

  G4AnalysisManager * analysis_manager = G4AnalysisManager::Instance();  

  analysis_manager->Write();
  analysis_manager->CloseFile();

  if (nofEvents == 0) 
     return;
}


}

