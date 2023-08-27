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
#include "TrackerSD.hh"
#include "G4HCofThisEvent.hh"
#include "G4Step.hh"
#include "G4ThreeVector.hh"
#include "G4SDManager.hh"
#include "G4ios.hh"
// #include <G4Electron.hh>
#include "G4ParticleTypes.hh"

namespace B2
{

TrackerSD::TrackerSD(const G4String& name, const G4String& hitsCollectionName) : G4VSensitiveDetector(name)
{
  // colletionName is a data member inherited from the parent class G4VSensitiveDetector
  collectionName.insert(hitsCollectionName);
}

// The initialize method is an overide method for the parent class G4VSensitiveDetector
void TrackerSD::Initialize(G4HCofThisEvent* hce)
{
  // Create hits collection, fHitCollection is a private data member of the child class TrackerSD (Sensetive Detector)
  fHitsCollection = new TrackerHitsCollection(SensitiveDetectorName, 
                                              collectionName[0]);

  // Add this collection in hce
  G4int hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  hce->AddHitsCollection( hcID, fHitsCollection );
}

// The ProcessHit method is an override method for the parenet class G4VSensitiveDetector
G4bool TrackerSD::ProcessHits(G4Step* aStep,
                                     G4TouchableHistory*)
{
  // energy deposit
  G4double edep = aStep->GetTotalEnergyDeposit();

  /**
    * When a particle enteres the sensitive volume, it has a beginning and an end point.
    * This is the point where the photon enteres the sensitive detector.
    */
  G4StepPoint * prestep_point = aStep->GetPreStepPoint();
  /**
    * Identify the type of particle.
    */
  G4ParticleDefinition* particle = aStep->GetTrack()->GetDefinition();
  if ( particle == G4Electron::Electron())
  {
     G4cout << G4endl << "---I am an electron ---" << G4endl;
  }
  if (particle == G4Positron::Positron())
  {
     G4cout << G4endl << "---I am a Positron ---" << G4endl;
  }
  if (particle == G4NeutrinoE::NeutrinoE())
  {
     G4cout << G4endl << "---I am a Neutrino ---" << G4endl;
  }

  if (particle == G4Gamma::Gamma())
  {
     G4cout << G4endl << "---I am a Gamma ---" << G4endl;
  }


  if (edep == 0.0) 
  {
    return false;
  }

  // The newHit is an intance of class TrackerHit.  The class inhertis from G4VHit
  auto newHit = new TrackerHit();

  newHit->SetTrackID  (aStep->GetTrack()->GetTrackID());
  newHit->SetChamberNb(aStep->GetPreStepPoint()->GetTouchableHandle()
                                               ->GetCopyNumber());
  newHit->SetEdep(edep);
  newHit->SetPos (aStep->GetPostStepPoint()->GetPosition());

  //The data member fHitsCollection is private data member of child class TrackerSD (Sensitive Detector)
  //It has a list of newHit's object.   
  fHitsCollection->insert( newHit );

  newHit->Print();
  return true;
}

//The EndOfEvent methos is an override method for the parent class G4VSensitiveDetector
void TrackerSD::EndOfEvent(G4HCofThisEvent*)
{
  if ( verboseLevel>1 ) 
  {
     G4int nofHits = fHitsCollection->entries();
     G4cout << G4endl
            << "-------->Hits Collection: in this event they are " 
            << nofHits
            << " hits in the tracker chambers: " << G4endl;
            
     for ( G4int i=0; i<nofHits; i++ ) (*fHitsCollection)[i]->Print();
  }
}

}

