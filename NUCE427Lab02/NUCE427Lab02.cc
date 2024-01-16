
#include "G4RunManagerFactory.hh"

#include "G4SteppingVerbose.hh"
#include "QBBC.hh"

#include "G4UImanager.hh"

#include "G4VisExecutive.hh"
#include "G4VisManager.hh"
#include "G4UIExecutive.hh"

#include "Randomize.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"

using namespace NUCE427LAB02;

/**
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * cmake -DCMAKE_BUILD_TYPE=RELEASE ..
 */

int main(int argc,
         char** argv)
{

  G4RunManager * run_manager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);

  G4UIExecutive* ui = nullptr;
  if ( argc == 1 ) 
  { 
      ui = new G4UIExecutive(argc, 
                             argv); 
  }

  // Initialize visualization
  G4VisManager* vis_manager = new G4VisExecutive;
  // G4VisExecutive can take a verbosity argument - see /vis/verbose guidance.
  vis_manager->Initialize();

  // Get the pointer to the User Interface manager
  G4UImanager* ui_manager = G4UImanager::GetUIpointer();

  G4int precision = 4;
  G4SteppingVerbose::UseBestUnit(precision);

  // Set mandatory initialization classes
  // Detector construction
  run_manager->SetUserInitialization(new DetectorConstruction()); 

  // Physics list
  // G4VModularPhysicsList* physics_list = new QBBC;
  G4VModularPhysicsList* physics_list = new PhysicsList(); 
  physics_list->SetVerboseLevel(1);
  run_manager->SetUserInitialization(physics_list);

  // User action initialization
  run_manager->SetUserInitialization(new ActionInitialization());

  // Initialize G4 kernel
  run_manager->Initialize();

  // Process macro or start UI session
  if ( ui ) 
  {
      // interactive mode
      ui_manager->ApplyCommand("/control/execute init_vis.mac");
      ui->SessionStart();
      delete ui;
  }
  else 
      {
        // batch mode
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        ui_manager->ApplyCommand(command+fileName);
      }

  // Job termination
  // delete physics_list;
  delete vis_manager;
  delete run_manager;
  return 0;
}
