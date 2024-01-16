//
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

#include "EventAction.hh"
#include "RunAction.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4AnalysisManager.hh"

namespace NUCE427LAB02
{

EventAction::EventAction(RunAction* run_action) : current_runaction(run_action)
{
  G4cout << "EventAction::EventAction" << G4endl;
  fenergy_deposite = 0.0;
}

EventAction::~EventAction()
{

}

void EventAction::BeginOfEventAction(const G4Event*)
{
  // In the begin of every event the energy desposite is cleared to zero.
  G4cout << "EventAction::BeginOfEventAction" << G4endl;
  fenergy_deposite = 0.0;
}

void EventAction::EndOfEventAction(const G4Event*)
{
  // accumulate statistics in run action
  // At the end of every event the energy desposite is cleared to zero.
  G4cout << "EventAction::EndOfEventAction" << G4endl;
  G4cout << "Energy Deposition " << fenergy_deposite << " MeV" << G4endl;
  G4AnalysisManager * analysis_manager = G4AnalysisManager::Instance();
  
  analysis_manager->FillNtupleDColumn(2, 0, fenergy_deposite);
  analysis_manager->AddNtupleRow(2);


}


}
