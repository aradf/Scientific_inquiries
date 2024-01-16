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
#include "SensitiveDetector.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4SystemOfUnits.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"

namespace NUCE427LAB02
{

SensitiveDetector::SensitiveDetector(const G4String& detector_name) : G4VSensitiveDetector(detector_name)
{
  // colletionName is a data member inherited from the parent class G4VSensitiveDetector
  G4int a = 5;
  G4cout << detector_name << G4endl;

}

SensitiveDetector::~SensitiveDetector()
{



}

// The ProcessHit method is an override method for the parenet class G4VSensitiveDetector
G4bool SensitiveDetector::ProcessHits(G4Step* step_of_track,
                                      G4TouchableHistory*)
{
  // energy deposit
  // G4double energy_deposited = step_of_track->GetTotalEnergyDeposit();

  // Get information about the track that enters the sensitive volume.
  G4Track * track_in_sensitivedetector_volume = step_of_track->GetTrack();

  track_in_sensitivedetector_volume->SetTrackStatus(fStopAndKill);

  // When a particle enteres the senstive volume, it has a pre and post step point.
  G4StepPoint * pre_step_point = step_of_track->GetPreStepPoint();
  G4StepPoint * post_step_point = step_of_track->GetPostStepPoint();

  G4ThreeVector position_photon = pre_step_point->GetPosition();

  // G4cout << "Photon position" << position_photon << G4endl; 

  // Get access to the detector position 
  // The touchable of logical volume.
  const G4VTouchable * touchable = step_of_track->GetPreStepPoint()->GetTouchable();

  G4int copy_number = touchable->GetCopyNumber();

  // G4cout << "copy_number = " << copy_number << G4endl; 

  G4VPhysicalVolume * physical_volume = touchable->GetVolume();
  G4ThreeVector position_detector = physical_volume->GetTranslation();
  G4ThreeVector momentum_photon = pre_step_point->GetMomentum();
  G4double wave_lenth = (1.239841939 * eV/momentum_photon.mag())*1E+03;

  G4cout << "Detector Position =  " << position_detector << G4endl;

  G4int event_number = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
  G4AnalysisManager * analysis_manager = G4AnalysisManager::Instance();
  
  analysis_manager->FillNtupleIColumn(0, 0, event_number);
  analysis_manager->FillNtupleDColumn(0, 1, position_photon[0]);
  analysis_manager->FillNtupleDColumn(0, 2, position_photon[1]);
  analysis_manager->FillNtupleDColumn(0, 3, position_photon[2]);
  analysis_manager->FillNtupleDColumn(0, 4, wave_lenth);
  analysis_manager->AddNtupleRow(0);

  analysis_manager->FillNtupleIColumn(1, 0, event_number);
  analysis_manager->FillNtupleDColumn(1, 1, position_detector[0]);
  analysis_manager->FillNtupleDColumn(1, 2, position_detector[1]);
  analysis_manager->FillNtupleDColumn(1, 3, position_detector[2]);
  analysis_manager->AddNtupleRow(1);

  return true;
}

}

