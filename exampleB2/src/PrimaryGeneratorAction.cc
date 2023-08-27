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

#include "PrimaryGeneratorAction.hh"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
//#include <G4GeneralParticleSource.hh>
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

namespace B2
{

PrimaryGeneratorAction::PrimaryGeneratorAction()
{
 
    G4int nofParticles = 1;
    fParticleGun = new G4ParticleGun(nofParticles);

    // default particle kinematic
    G4ParticleDefinition* particleDefinition = G4ParticleTable::GetParticleTable()->FindParticle("proton");

    fParticleGun->SetParticleDefinition(particleDefinition);
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.0,
                                                             0.0,
                                                             1.0));
    fParticleGun->SetParticleEnergy(3.0*GeV);
 
    /*
    fGPS = new G4GeneralParticleSource();
    G4ParticleDefinition * myParticle;
    myParticle = G4ParticleTable::GetParticleTable()->FindParticle("e-");
    fGPS->SetParticleDefinition(myParticle);
    fGPS->GetCurrentSource()->GetEneDist()->SetMonoEnergy(100.*MeV);
    fGPS->GetCurrentSource()->GetAngDist()->SetParticleMomentumDirection(G4ThreeVector(1.,0.,0.));
    fGPS->GetCurrentSource()->GetPosDist()->SetCentreCoords(G4ThreeVector(0,0,1));    
    */

}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
    //delete fGPS;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
    // This function is called at the begining of event

    // In order to avoid dependence of PrimaryGeneratorAction
    // on DetectorConstruction class we get world volume
    // from G4LogicalVolumeStore.
    G4double worldZHalfLength = 0;
    G4LogicalVolume* worldLV = G4LogicalVolumeStore::GetInstance()->GetVolume("World");
    G4Box* worldBox = nullptr;
    if ( worldLV ) 
    {
        worldBox = dynamic_cast<G4Box*>(worldLV->GetSolid());
    }
    
    if ( worldBox ) 
    { 
        worldZHalfLength = worldBox->GetZHalfLength();
    }
    else  
    {
      G4cerr << "World volume of box not found." << G4endl;
      G4cerr << "Perhaps you have changed geometry." << G4endl;
      G4cerr << "The gun will be place in the center." << G4endl;
    }

    ////////////////////////////
    G4double x0 = 0.0 * cm, y0 = 0.0 * cm;
    G4double dx0 = 2.0 * cm, dy0 = 2.0 * cm;
    x0 += dx0*(G4UniformRand()-0.5);
    y0 += dy0*(G4UniformRand()-0.5);
    ////////////////////////////

    // Note that this particular case of starting a primary particle on the world boundary
    // requires shooting in a direction towards inside the world.
    // fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., -worldZHalfLength));

    fParticleGun->SetParticlePosition(G4ThreeVector( x0, y0, -worldZHalfLength));
    fParticleGun->GeneratePrimaryVertex(anEvent);

    /*
    fGPS->SetParticlePosition(G4ThreeVector( x0, y0, -worldZHalfLength));
    fGPS->GeneratePrimaryVertex(anEvent);
    */

}

}

