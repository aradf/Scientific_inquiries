#include <iostream>
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4VisManager.hh"
#include "G4UIExecutive.hh"

#include "RadioactiveDecayDetectorConstruncion.hh"
#include "RadioactiveDecayPhysicList.hh"
#include "RadioactiveDecayActionInitialization.hh"

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

   run_manager->SetUserInitialization( new CRadioactiveDecayDetectorConstruction());
   run_manager->SetUserInitialization( new CRadioactiveDecayPhysicsList());
   run_manager->SetUserInitialization( new CRadioactiveDecayActionInitialization());

   /** 
    * initialize G4 kernel 
    */
   run_manager->Initialize();

   /**
    * The Class G4UIExectuive is a convenient way for choosing a session type.  
    * A user can instantiated an object of the G4UIExecutive type by passing 
    * a session at the construction level like "G4UIExecutive(argc, argv, "qt");"
    */
   G4UIExecutive * ui_executive = 0;
   if (argc == 1)
   {
      ui_executive = new G4UIExecutive(argc, argv);
   }

   /**
    * The G4VisExecutive and G4VisManager classes register, manage and control
    * the visualization procedures (graphical system) of the Geant4 Sim Application.
    */
   G4VisManager * visual_manager = new G4VisExecutive();
   visual_manager->Initialize();

   /**
    * Get the pointer to the UI manager and set verbosities of various Geant4 manager 
    * classes.  
    */
   G4UImanager * ui_manager = G4UImanager::GetUIpointer();

   if (! ui_executive)
   {
      /*
       * Batch mode
       */
      G4String command = "/control/execute ";
      G4String file_name = command + argv[1];
      ui_manager->ApplyCommand(file_name);
   }
   else {
      /*
       * Interactive mode.
       */
      // ui_manager->ApplyCommand("/control/execute visualize.mac");
      ui_manager->ApplyCommand("/vis/open OGL");
      ui_manager->ApplyCommand("/vis/viewer/set/viewpointVector 1 1 1");
      ui_manager->ApplyCommand("/vis/drawVolume");
      ui_manager->ApplyCommand("/vis/viewer/set/autoRefresh true");
      ui_manager->ApplyCommand("/vis/scene/add/trajectories smooth");

      ui_executive->SessionStart();
   }

   delete run_manager;
   delete ui_executive;

   return 0;
}
