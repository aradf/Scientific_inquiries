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
#include "PrimaryGeneratorAction.hh"

#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4AccumulableManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

using namespace B3;

namespace B3a
{

RunAction::RunAction()
{
      //add new units for dose
      const G4double milligray = 1.e-3*gray;
      const G4double microgray = 1.e-6*gray;
      const G4double nanogray  = 1.e-9*gray;
      const G4double picogray  = 1.e-12*gray;

      new G4UnitDefinition("milligray", 
                           "milliGy" , 
                           "Dose", 
                           milligray);

      new G4UnitDefinition("microgray", 
                           "microGy" , 
                           "Dose", 
                           microgray);
                           
      new G4UnitDefinition("nanogray" , 
                           "nanoGy"  , 
                           "Dose", 
                           nanogray);
                           
      new G4UnitDefinition("picogray" , 
                           "picoGy"  , 
                           "Dose", 
                           picogray);

      // Register accumulable to the accumulable manager
      // Return an instance of the Accumulable manager.
      // Register the two vecotrs data type of accumulable objects.
      // Accumulale means something that could increase gradually.
      G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();
      accumulableManager->RegisterAccumulable(fGoodEvents);
      accumulableManager->RegisterAccumulable(fSumDose);
}

void RunAction::BeginOfRunAction(const G4Run* run)
{
  G4cout << "### Run " << run->GetRunID() << " start." << G4endl;

  // reset accumulables to their initial values
  G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Reset();

  //inform the runManager to save random number seed
  G4RunManager::GetRunManager()->SetRandomNumberStore(false);
}

void RunAction::EndOfRunAction(const G4Run* current_run)
{
  // return the number of events in the current run.
  G4int nofEvents = current_run->GetNumberOfEvent();
  if (nofEvents == 0) 
     return;

  // Merge accumulables like 
  G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Merge();

  // Run conditions
  //  note: There is no primary generator action object for "master"
  //        run manager for multi-threaded mode.
  const auto generatorAction = static_cast<const PrimaryGeneratorAction*>(G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction());

  G4String particle_name;
  if (generatorAction)
  {
    G4ParticleDefinition* particle = generatorAction->GetParticleGun()->GetParticleDefinition();
    particle_name = particle->GetParticleName();
  }

  // Print results
  if (IsMaster())
  {
    G4cout
     << G4endl
     << "--------------------End of Global Run-----------------------"
     << G4endl
     << "  The run was " << nofEvents << " events for " << particle_name;
  }
  else
  {
    G4cout
     << G4endl
     << "--------------------End of Local Run------------------------"
     << G4endl
     << "  The run was " << nofEvents << " events for " << particle_name;
  }
  G4cout
     << "; Numberb of 'good' e+ annihilations: " << fGoodEvents.GetValue()  << G4endl
     << " Total dose in patient : " << G4BestUnit(fSumDose.GetValue(),"Dose")
     << G4endl
     << "------------------------------------------------------------" << G4endl
     << G4endl;
}

}

