#include "G4RunManagerFactory.hh"
#include "G4PhysListFactory.hh"
#include "G4UImanager.hh"
#include "G4VisManager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"

#include "G4EmStandardPhysics_option4.hh"

#include "CFirstStepDetectorConstruction.hh"
#include "CFirstStepActionInitialization.hh"

#include <globals.hh>

/**
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * cmake -DCMAKE_BUILD_TYPE=RELEASE ..
 */

int main (int argc, char** argv)
{
   G4String macro_file = "";
   G4UIExecutive * ui_executive = nullptr;
   if (argc == 1)
   {
      /* 
       * There is no input argument; Must have an interactive session.
       */
      ui_executive = new G4UIExecutive(argc, argv);
      // ui_executive = new G4UIExecutive(argc, argv, "tcsh");
      /*
       * Example of command lines
       * /tracking/verbose 1
       * /run/initialize
       * /gun/energy 5 MeV
       * /gun/particle gamma
       * /run/beamOn 100
       * /vis/list
       * /vis/drawVolume
       * /vis/open OGLI
       * exit
       */

   }
   else
      {
         // There is macro file.
         macro_file = argv[1];
      }

   G4RunManager *run_manager = G4RunManagerFactory::CreateRunManager();
   
   // set mandatory initialization classes
   FS::CFirstStepDetectorConstruction * detector_construction = new FS::CFirstStepDetectorConstruction();
   run_manager->SetUserInitialization(detector_construction);

   // const G4String physics_listName = "FTFP_BERT";
   const G4String physics_listName = "FTFP_BERT_EMZ";
   // const G4String physics_listName = "QGSP_BIC_HP_EMZ";
   // const G4String physics_listName = "QBBC";
   // G4PhysListFactory physics_listFactory; 
   // G4VModularPhysicsList * physics_list = physics_listFactory.GetReferencePhysList( physics_listName );
   // run_manager->SetUserInitialization ( physics_list );

   // const G4String physics_listName = "G4EmPenelopePhysics";
   G4PhysListFactory physics_listFactory; 
   G4VModularPhysicsList * physics_list = physics_listFactory.GetReferencePhysList( physics_listName );
   run_manager->SetUserInitialization ( physics_list );

   FS::CFirstStepActionInitialization * action_initialization = new FS::CFirstStepActionInitialization ( detector_construction );
   run_manager->SetUserInitialization( action_initialization );

   /*
    * Set up visualization and initialize.
    */
   G4VisManager * visual_manager = new G4VisExecutive();
   visual_manager->Initialize();

   /*
    * 1. Initialize the run manager.
    * 2. Simualte two events using BeamOn method.
    */
   G4UImanager * ui_manager = G4UImanager::GetUIpointer();
   if ( ui_executive == nullptr)    
   {
      // Macro file exists:  execute the file.
      G4String command = "/control/execute ";
      ui_manager->ApplyCommand(command + macro_file);
   }
   else
      {
         // interactive: start a session.
         ui_executive->SessionStart();
         delete ui_executive;
         ui_executive = nullptr;
      }

   // run_manager->Initialize();
   // ui_manager->ApplyCommand("/tracking/verbose 1");
   // run_manager->BeamOn(1);

   // G4cout << G4endl;
   // G4cout << "************ Second Run ****************";
   // G4cout << G4endl;

   // detector_construction->set_targetThickness(2.0*CLHEP::cm);
   // run_manager->Initialize();
   // run_manager->BeamOn(1);

   delete run_manager;
   run_manager = nullptr;
   return 0;

}