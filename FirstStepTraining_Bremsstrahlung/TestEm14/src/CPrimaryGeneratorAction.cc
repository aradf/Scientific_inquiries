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
#include "CPrimaryGeneratorAction.hh"
#include "CDetectorConstruction.hh"

#include "G4Event.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

CPrimaryGeneratorAction::CPrimaryGeneratorAction(CDetectorConstruction* detector_constructor)
                                                :G4VUserPrimaryGeneratorAction(),
                                                fparticle_gun(nullptr),
                                                fdetector_constructor(detector_constructor)
{
   fparticle_gun  = new G4ParticleGun( 1 );
   G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle( "gamma" );
   G4cout << "INFO: Particle - "
          << particle->GetParticleName()
          << G4endl;
   fparticle_gun->SetParticleDefinition( particle );
   fparticle_gun->SetParticleEnergy( 1 * MeV );    
   fparticle_gun->SetParticleMomentumDirection(G4ThreeVector(1.0,
                                                             0.0,
                                                             0.0));
}

CPrimaryGeneratorAction::~CPrimaryGeneratorAction()
{
  delete fparticle_gun;
}

void CPrimaryGeneratorAction::GeneratePrimaries(G4Event* an_event)
{
  /*
   * This function is called at the begining of event
   */
  G4double half_size = 0.5 * ( fdetector_constructor->get_size() );
  G4double x0 = - half_size;

  /*
   * randomize (y0,z0)
   */
  G4double beam = 0.8 * half_size; 
  G4double y0 = ( 2 * G4UniformRand()-1.0 ) * beam;
  G4double z0 = ( 2 * G4UniformRand()-1.0 ) * beam;

  // G4cout << "INFO: Primary Particle Positions "
  //        << "(x0, y0, z0) = " 
  //        << x0
  //        << ", "
  //        << y0
  //        << ", "
  //        << z0
  //        << " "
  //        << G4endl;


  fparticle_gun->SetParticlePosition(G4ThreeVector( x0,
                                                    y0,
                                                    z0 ));
  fparticle_gun->GeneratePrimaryVertex( an_event );
}

