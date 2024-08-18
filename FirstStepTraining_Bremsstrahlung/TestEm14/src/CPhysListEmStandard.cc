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

#include "CPhysListEmStandard.hh"

#include "G4BuilderType.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"

#include "G4RayleighScattering.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4ComptonScattering.hh"
#include "G4KleinNishinaModel.hh"
#include "G4GammaConversion.hh"
#include "G4GammaConversionToMuons.hh"

#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4MuIonisation.hh"
#include "G4MuBremsstrahlung.hh"
#include "G4MuPairProduction.hh"

#include "G4hIonisation.hh"
#include "G4hBremsstrahlung.hh"
#include "G4hPairProduction.hh"

#include "G4ionIonisation.hh"
#include "G4IonParametrisedLossModel.hh"

#include "G4LossTableManager.hh"
#include "G4UAtomicDeexcitation.hh"

#include "G4SystemOfUnits.hh"

CPhysListEmStandard::CPhysListEmStandard(const G4String& name)
                                      : G4VPhysicsConstructor(name)
{
  G4EmParameters* electromagnetic_param = G4EmParameters::Instance();
  electromagnetic_param->SetDefaults();
  electromagnetic_param->SetMinEnergy( 10 * eV );
  electromagnetic_param->SetMaxEnergy( 10 * TeV );
  electromagnetic_param->SetNumberOfBinsPerDecade( 10 );
  
  electromagnetic_param->SetVerbose( 0 );
  electromagnetic_param->Dump();
}

CPhysListEmStandard::~CPhysListEmStandard()
{}

void CPhysListEmStandard::ConstructProcess()
{ 
  /*
   * Add Standard Electro Mangnetism (EM) Processes.
   */
  auto particle_iterator = GetParticleIterator();
  particle_iterator->reset();
  while( ( *particle_iterator )() )
  {
     G4ParticleDefinition* particle = particle_iterator->value();
     G4ProcessManager* process_manager = particle->GetProcessManager();
     G4String particle_name = particle->GetParticleName();

   //   G4cout << "INFO: Particle Name: "
   //          << particle_name 
   //          << G4endl;
     
     if (particle_name == "gamma") 
     {

        ////process_manager->AddDiscreteProcess(new G4RayleighScattering);               
        process_manager->AddDiscreteProcess(new G4PhotoElectricEffect);
        G4ComptonScattering* compt   = new G4ComptonScattering;
        compt->SetEmModel(new G4KleinNishinaModel());
        process_manager->AddDiscreteProcess(compt);
        process_manager->AddDiscreteProcess(new G4GammaConversion);
        process_manager->AddDiscreteProcess(new G4GammaConversionToMuons);  
     
     } 
     else if (particle_name == "e-") 
     {
        process_manager->AddProcess(new G4eIonisation,        -1,-1,1);
        process_manager->AddProcess(new G4eBremsstrahlung,    -1,-1,2);
            
     } 
     else if (particle_name == "e+") 
     {
        process_manager->AddProcess(new G4eIonisation,        -1,-1,1);
        process_manager->AddProcess(new G4eBremsstrahlung,    -1,-1,2);
        process_manager->AddProcess(new G4eplusAnnihilation,   0,-1,3);
    } 
    else if (particle_name == "mu+" || 
             particle_name == "mu-" ) 
    {
       process_manager->AddProcess(new G4MuIonisation,      -1,-1,1);
       process_manager->AddProcess(new G4MuBremsstrahlung,  -1,-1,2);
       process_manager->AddProcess(new G4MuPairProduction,  -1,-1,3);       
    } 
    else if( particle_name == "proton" ||
               particle_name == "pi-"  ||
               particle_name == "pi+"  ) 
    {
       process_manager->AddProcess(new G4hIonisation,       -1,-1,1);
       process_manager->AddProcess(new G4hBremsstrahlung,   -1,-1,2);      
       process_manager->AddProcess(new G4hPairProduction,   -1,-1,3);        
    } 
    else if( particle_name == "alpha" || 
             particle_name == "He3" ) 
    {
       process_manager->AddProcess(new G4ionIonisation,     -1,-1,1);

    } 
    else if( particle_name == "GenericIon" ) 
    {
       G4ionIonisation* ionIoni = new G4ionIonisation();
       ionIoni->SetEmModel(new G4IonParametrisedLossModel());
       process_manager->AddProcess(ionIoni,                 -1,-1,1);
    } 
    else if ((!particle->IsShortLived()) &&
             (particle->GetPDGCharge() != 0.0) && 
             (particle->GetParticleName() != "chargedgeantino")) 
    {
      //all others charged particles except geantino
      process_manager->AddProcess(new G4hIonisation,       -1,-1,1);      
    }
  }

  /*
   * Deexcitation
   */    
  G4VAtomDeexcitation* deex = new G4UAtomicDeexcitation();
  G4LossTableManager::Instance()->SetAtomDeexcitation(deex);
}
