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

#include "CRunAction.hh"
#include "CDetectorConstruction.hh"
#include "CPrimaryGeneratorAction.hh"
#include "CHistogramManager.hh"
#include "CRun.hh"

#include "G4Run.hh"
#include "G4UnitsTable.hh"
#include "G4EmCalculator.hh"

#include "Randomize.hh"
#include <iomanip>

CRunAction::CRunAction(CDetectorConstruction* detector_constructor, 
                     CPrimaryGeneratorAction* kinetic_energy)
                     :G4UserRunAction(),
                     fdetector_constructor(detector_constructor),
                     fprimary_generator(kinetic_energy),
                     fgenerated_run(nullptr),
                     fhistogram_manager(nullptr)
{ 
   fhistogram_manager = new CHistogramManager();
}

CRunAction::~CRunAction()
{
   delete fhistogram_manager;
}

G4Run* CRunAction::GenerateRun()
{ 
   fgenerated_run = new CRun(fdetector_constructor); 
   return fgenerated_run;
}

void CRunAction::BeginOfRunAction(const G4Run*)
{    
   /*
    * show Rndm status
    */
   if ( isMaster )  
      G4Random::showEngineStatus();
     
   /*
    * keep run condition
    */
   if ( fprimary_generator ) 
   { 
      G4ParticleDefinition* particle = fprimary_generator->get_particleGun()->GetParticleDefinition();
      G4double energy = fprimary_generator->get_particleGun()->GetParticleEnergy();
      fgenerated_run->set_primary( particle, 
                                   energy);
   }    
      
   /*
    * histograms
    */
   G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();
   if ( analysisManager->IsActive() ) 
   {
      analysisManager->OpenFile();
   }   
}

void CRunAction::EndOfRunAction(const G4Run*)
{
   /*
    * Compute and print statistic 
    */
   if ( isMaster ) 
      fgenerated_run->end_ofSingleRun();
             
   /*
    * save histograms
    */
   G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();  
   if ( analysis_manager->IsActive() ) 
   {
      analysis_manager->Write();
      analysis_manager->CloseFile();
   }    
  
   /*
    * show Rndm status
    */
   if ( isMaster )  
      G4Random::showEngineStatus(); 
}

