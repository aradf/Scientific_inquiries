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
#include "CSteppingAction.hh"
#include "CRun.hh"
#include "CHistogramManager.hh"
#include "G4RunManager.hh"

CSteppingAction::CSteppingAction() : G4UserSteppingAction()
{ 

}

CSteppingAction::~CSteppingAction()
{ 

}

void CSteppingAction::UserSteppingAction(const G4Step* current_step)
{
   /*
    * current_step is an instance of G4Step and provides the information regarding the 
    * change in the state of the particle (that is under tracking) within a simulation step.
    * 
    */

   /*
    * The two pre-step and post-step points store information (position, direction, energy, 
    * material, volume, etc…) that belong to the corresponding point (space/time/step)
    */
    const G4StepPoint* end_point = current_step->GetPostStepPoint();
    G4String process_name = end_point->GetProcessDefinedStep()->GetProcessName();
   //  G4cout << "INFO: process_name: "
   //         << process_name
   //         << G4endl;

   /*
    * Explicite type conversion 'static casting'.  Converts a pointer of some object to 
    * pointer variable of related class type CRun.  The static casting is down/up.  The 
    * GetNonConstCurrentRun returns a pointer variable of parent class type G4Run.
    */
   CRun* current_run = static_cast< CRun * > ( G4RunManager::GetRunManager()->GetNonConstCurrentRun() );
  
   /*
    * A particle is physically stands on the boundary (the step status of the post step point i.e.
    * G4Step::GetPostStepPoint()->GetStepStatus() is fGeomBoundary )
    */
   G4bool transmit = (end_point->GetStepStatus() <= fGeomBoundary);  
   if ( transmit ) 
   { 
      current_run->count_processes( process_name ); 
   }
   else 
   {                         
      /*
       * count real processes and sum track length 
       */
      G4double step_length = current_step->GetStepLength();
      current_run->count_processes( process_name );  
      current_run->sum_tracks( step_length );
   }
  
   /*
    * plot final state
    */
   G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();
     
   /*
    * scattered primary particle
    */
   G4int id = 1;
   if (current_step->GetTrack()->GetTrackStatus() == fAlive) 
   {
      G4double energy = end_point->GetKineticEnergy();      
      analysis_manager->FillH1( id , 
                                energy);

      id = 2;
      G4ThreeVector direction = end_point->GetMomentumDirection();
      G4double cos_theta = direction.x();
      analysis_manager->FillH1( id , 
                                cos_theta );     
   }  
  
   /*
    * Get secondaries in current step.  
    * Standard Tempalte Library (STL)'s vector container holdes const G4Track pointers.  
    * 'secndary' is a pointer variable == 0x1234, '*' operator
    * has the content of the secondaryInCurrentStep object;  &secondary == 0xABCD; 
    */
   const std::vector< const G4Track* > *secondary = current_step->GetSecondaryInCurrentStep();
   size_t number_secondaries = (*secondary).size();
   for (size_t idx_count = 0; idx_count < (*secondary).size(); idx_count++) 
   {
      /*
       * Particle Data Group (PDG) has fundamental charge of the particle.
       */
      G4double charge = (*secondary)[idx_count]->GetDefinition()->GetPDGCharge();
      if (charge != 0.0) 
      { 
        id = 3; 
      } 
      else 
      { 
        id = 5; 
      }
      G4double energy = (*secondary)[idx_count]->GetKineticEnergy();
      /*
       * Fill one dimensinoal Histogram.
       */
      analysis_manager->FillH1( id , 
                                energy );

      ++id;
      G4ThreeVector direction = (*secondary)[idx_count]->GetMomentumDirection();      
      G4double cos_theta = direction.x();
      analysis_manager->FillH1( id ,
                                cos_theta );
      /*
      * energy tranferred to charged secondaries  
      */
      if (charge != 0.0) 
      { 
         current_run->sum_energyTransfered( energy ); 
      }          
   }
         
   /*
    * kill event after first interaction
    */
   G4RunManager::GetRunManager()->AbortEvent();  
}



