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
//
//
#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"

#include "G4Step.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4LogicalVolume.hh"

namespace NUCE427LAB02
{

SteppingAction::SteppingAction(EventAction* transfered_eventaction): recieved_eventaction(transfered_eventaction)
{
   G4cout << "SteppingAction::SteppingAction" << G4endl;
}

SteppingAction::~SteppingAction()
{

}

void SteppingAction::UserSteppingAction(const G4Step* current_step)
{
   // get volume of the current step
   G4LogicalVolume * current_volume = current_step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
   const DetectorConstruction  * detector_construction = static_cast<const DetectorConstruction*>(G4RunManager::GetRunManager()->GetUserDetectorConstruction());
   G4LogicalVolume * current_scoringvolume = detector_construction->get_scoring_volume();
   if (current_scoringvolume != current_volume)
      return;

   G4double energy_deposition_in_step = current_step->GetTotalEnergyDeposit();
   G4cout << "Total Energy Deposited from SteppingAction::UserSteppingAction " << energy_deposition_in_step << G4endl;
   recieved_eventaction->add_energy_deposite(energy_deposition_in_step);

}

}
