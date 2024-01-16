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


#include "G4RunManager.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ChargedGeantino.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "PrimaryGeneratorAction.hh"
#include "G4Geantino.hh"

namespace NUCE427LAB02
{

PrimaryGeneratorAction::PrimaryGeneratorAction()
{
  // Number of particles per event.
  G4int n_particle = 1;
  current_particle_gun  = new G4ParticleGun(n_particle);

  G4ParticleTable* particle_table = G4ParticleTable::GetParticleTable();
  G4String particle_name = "geantino";
  G4ParticleDefinition* particle = particle_table->FindParticle(particle_name);

  G4ThreeVector particle_position(0.0, 0.0, 0.0);
  G4ThreeVector particle_momentum(0.0, 0.0, 1.0);

  current_particle_gun->SetParticlePosition(particle_position);
  current_particle_gun->SetParticleMomentumDirection(particle_momentum);
  current_particle_gun->SetParticleMomentum(0.0 * GeV);
  current_particle_gun->SetParticleDefinition(particle);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete current_particle_gun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* transfered_event)
{
  G4ParticleDefinition * particle = current_particle_gun->GetParticleDefinition();
  if (particle == G4Geantino::Geantino())
  {
    // Au-198
    G4int Z = 79;
    G4int A = 198;

    G4double particle_charge = 0.0 * eplus;
    G4double particle_energy = 0.0 * keV;
    G4ParticleDefinition * ion = G4IonTable::GetIonTable()->GetIon(Z, A, particle_energy);
    current_particle_gun->SetParticleDefinition(ion);
    current_particle_gun->SetParticleCharge(particle_charge);
  }

  //create vertex
  current_particle_gun->GeneratePrimaryVertex(transfered_event);
}

}


