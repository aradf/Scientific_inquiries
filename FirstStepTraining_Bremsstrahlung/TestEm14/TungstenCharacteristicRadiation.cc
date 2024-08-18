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

#include "G4Types.hh"
#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4SteppingVerbose.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"

#include "Randomize.hh"

#include "CDetectorConstruction.hh"
#include "CPhysicsList.hh"
#include "CActionInitialization.hh"

/**
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * cmake -DCMAKE_BUILD_TYPE=RELEASE ..
 */

int main(int argc,char** argv) 
{

  /*
   * detect interactive mode (if no arguments) and define UI session
   */
  G4UIExecutive* ui_executive = nullptr;
  if (argc == 1) 
  {
    ui_executive = new G4UIExecutive(argc,argv);
  }

  /*
   * choose the Random engine 
   */
  G4Random::setTheEngine(new CLHEP::RanecuEngine);

  /*
   * Use SteppingVerbose with Unit 
   */
  G4int precision = 4;
  G4SteppingVerbose::UseBestUnit( precision );
  
  /*
   * Creating run manager.  
   * 1. G4RunManager is responsible to control the flow of run (top level simulaition unit) 
   * 2. Initialization of the run like building, Simulation environment.
   * 3. 
   */
  G4RunManager * run_manager = G4RunManagerFactory::CreateRunManager();
    
  if ( argc == 3 ) 
  { 
     G4int number_threads = G4UIcommand::ConvertToInt( argv[2] );
     run_manager->SetNumberOfThreads( number_threads );
  }

  /*
   * Set mandatory initialization classes.
   * These are the components of a simulatin, like geometry, physics and the primary particle generation
   * setings that change from one problem to the next.
   */
  CDetectorConstruction* detector_construction = new CDetectorConstruction;
  run_manager->SetUserInitialization( detector_construction );
  run_manager->SetUserInitialization( new CPhysicsList );
  run_manager->SetUserInitialization( new CActionInitialization( detector_construction ) );

  /*
   * initialize visualization
   */
  G4VisManager* visualization_manager = nullptr;

  /*
   * get the pointer to the User Interface manager
   */
  G4UImanager* ui_manager = G4UImanager::GetUIpointer();

  if ( ui_executive )  
  {
      //interactive mode
      visualization_manager = new G4VisExecutive;
      visualization_manager->Initialize();
      ui_executive->SessionStart();
      delete ui_executive;
  }
  else  
  {
      //batch mode
      G4String command = "/control/execute ";
      G4String fileName = argv[1];
      ui_manager->ApplyCommand( command + fileName );
  }

  //job termination
  delete visualization_manager;
  delete run_manager;
}


