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
/// \file B1/src/PrimaryGeneratorAction.cc
/// \brief Implementation of the B1::PrimaryGeneratorAction class

#include "PrimaryGeneratorAction.hh"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4RunManager.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

namespace B1
{


PrimaryGeneratorAction::PrimaryGeneratorAction()
{
  G4int n_particle = 1;
  fParticleGun  = new G4ParticleGun(n_particle);
  //fParticleGps  = new G4GeneralParticleSource();

  // default particle kinematic
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  // Particle type is a gamma
  // Momentum has the direction of 0, 0, and 1.0
  // The Kinertic energy of gamma is 6 MeV
  // G4ParticleDefinition* particle = particleTable->FindParticle(particleName="gamma");
  G4ParticleDefinition* particle = particleTable->FindParticle(particleName="proton");
  fParticleGun->SetParticleDefinition(particle);
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.0,
                                                           0.0,
                                                           1.0));
  fParticleGun->SetParticleEnergy(6.0 * MeV);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
  //delete fParticleGps;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  //this function is called at the begining of ecah event
  //

  // In order to avoid dependence of PrimaryGeneratorAction
  // on DetectorConstruction class we get Envelope volume
  // from G4LogicalVolumeStore.

  G4double envSizeXY = 0;
  G4double envSizeZ = 0;
  
  // fEnvelopBox is a pointer variable of class type G4Box.  It is instantiated to a null pointer.
  // The not null return a true the first time.
  if (!fEnvelopeBox)
  {
    G4LogicalVolume* envLV = G4LogicalVolumeStore::GetInstance()->GetVolume("Envelope");
    
    if ( envLV ) 
    {
       fEnvelopeBox = dynamic_cast<G4Box*>(envLV->GetSolid());
    }
  }

  if ( fEnvelopeBox ) 
  {
     envSizeXY = fEnvelopeBox->GetXHalfLength()*2.0;
     envSizeZ = fEnvelopeBox->GetZHalfLength()*2.0;
  }
  else  {
           G4ExceptionDescription msg;
           msg << "Envelope volume of box shape not found.\n";
           msg << "Perhaps you have changed geometry.\n";
           msg << "The gun will be place at the center.";
           G4Exception("PrimaryGeneratorAction::GeneratePrimaries()",
                       "MyCode0002",
                       JustWarning,
                       msg);
        }

  // A uniform Randum Number is a randum number where each possible number
  // in the range is just as likely as any other possible number.  Every number
  // has the same chance of being selected with no bias or pattern in the selection
  // process.  Momentum, direction and Kinetic energy can also be randomised.
  G4double size = 0.1;
  G4double tmpG4UniformRandx0 = G4UniformRand();
  G4double tmpG4UniformRandy0 = G4UniformRand();
  G4double x0 = size * envSizeXY * (tmpG4UniformRandx0-0.5);
  G4double y0 = size * envSizeXY * (tmpG4UniformRandy0-0.5);
  G4double z0 = -0.5 * envSizeZ;

  // The X and Y position of the particle has a uniform random distribution.
  fParticleGun->SetParticlePosition(G4ThreeVector(x0,
                                                  y0,
                                                  z0));
  fParticleGun->GeneratePrimaryVertex(anEvent);
}

}


