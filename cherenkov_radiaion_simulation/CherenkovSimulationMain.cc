#include <iostream>
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4VisManager.hh"
#include "G4UIExecutive.hh"

#include "CherenkovActionInitialization.hh"
#include "CherenkovDetectorConstruction.hh"
#include "CherenkovPhysicList.hh"

/**
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * cmake -DCMAKE_BUILD_TYPE=RELEASE ..
 */

int main(int argc, char** argv)
{
   /** 
    * Construct the default run manager.  All feature of Geant4/MonteCarlo 
    * run is managed in this class. Features like particle history, tracking, 
    * scoring, physics, detectores, etc..
    */
   G4RunManager * run_manager = new G4RunManager();
   
   run_manager->SetUserInitialization( new CCherenkovPhysicsList());
   run_manager->SetUserInitialization( new CCherenkovDetectorConstruction());
   run_manager->SetUserInitialization( new CCherenkovActionInitialization());

   /** 
    * initialize G4 kernel 
    */
   run_manager->Initialize();

   /**
    *
    */
   G4UIExecutive * ui_executive = 0;
   if (argc == 1)
   {
      ui_executive = new G4UIExecutive(argc, argv);
   }

   /**
    *
    */
   G4VisManager * visual_manager = new G4VisExecutive();
   visual_manager->Initialize();

   /**
    * get the pointer to the UI manager and set verbosities
    */
   G4UImanager * ui_manager = G4UImanager::GetUIpointer();

   if (ui_executive)
   {
      ui_manager->ApplyCommand("/control/execute visualize.mac");

      ui_executive->SessionStart();
   }
   else
   {
      G4String command = "/control/execute ";
      G4String file_name = command + argv[1];
      ui_manager->ApplyCommand(file_name);
   }

   delete run_manager;
   delete ui_executive;

   return 0;
}
