#include <G4RunManagerFactory.hh>
#include <G4UIExecutive.hh>
#include <G4VisExecutive.hh>
#include <G4UImanager.hh>
#include <G4VisManager.hh>

#include <globals.hh>

#include <CLab04DetectorConstruction.hh>
#include <CLab04PhysicsList.hh>
#include <CLab04ActionInitialization.hh>

using namespace NUCE427LAB04;

/*
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * cmake -DCMAKE_BUILD_TYPE=RELEASE ..
 * root output.root
 * new TBrowser
 * .q
 */

int main( int argc, 
          char** argv )
{
   G4cout << "**************************************************************"
          << G4endl
          << "                        NUCE427Lab04                          "
          << G4endl
          << "**************************************************************"
          << G4endl;

   G4String macro_file = "";

   G4RunManager* run_manager = new G4RunManager();
   run_manager->SetUserInitialization(new CLab04DetectorConstructor());
   run_manager->SetUserInitialization(new CLab04PhysicsList());
   run_manager->SetUserInitialization(new CLab04ActionInitialization());

   /*
    * move the run_manager's initialization to the mac file.
    */
   // run_manager->Initialize();

   G4UIExecutive* ui_executive = nullptr;
   if (argc == 1)
   {
       ui_executive = new G4UIExecutive( argc, 
                                         argv );
   }
    
   G4VisManager* visualize_manager = new G4VisExecutive();
   visualize_manager->Initialize();

   G4UImanager* ui_manager = G4UImanager::GetUIpointer();
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
   if (ui_executive)
   {
       delete ui_executive;
   }

  return 0;
}