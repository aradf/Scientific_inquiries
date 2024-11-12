#include <G4RunManagerFactory.hh>
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include "G4UImanager.hh"
#include "G4PhysListFactory.hh"
#include "G4HadronPhysicsFTFP_BERT_HP.hh"

#include <globals.hh>
#include "CLab03DetectorConstruction.hh"
#include "CLab03PhysicsList.hh"
#include "CLab03ActionInitialization.hh"

using namespace NUCE427LAB03;

/*
 * cmake -DCMAKE_BUILD_TYPE=DEBUG ..
 * cmake -DCMAKE_BUILD_TYPE=RELEASE ..
 */

int main( int argc, 
          char** argv )
{
   G4cout << "**************************************************************"
          << G4endl
          << "                        NUCE427Lab03                          "
          << G4endl
          << "**************************************************************"
          << G4endl;

   G4String macro_file = "";
   G4UIExecutive* ui_executive = nullptr;
   if (argc == 1)
   {
      /* 
       * There is no input argument; Must have an interactive session.
       */
      ui_executive = new G4UIExecutive(argc, argv);
      // ui_executive = new G4UIExecutive(argc, argv, "tcsh");
   }
   else
      {
         // There is macro file.
         macro_file = argv[1];
      }

   G4RunManager* run_manager = G4RunManagerFactory::CreateRunManager();

   /*
    * Set mandatory initialization classes.
    */
   NUCE427LAB03::CLab03DetectorConstructor* detector_construction = new NUCE427LAB03::CLab03DetectorConstructor();
   run_manager->SetUserInitialization( detector_construction );
   
   /*
    * TBD: Adjust the physics List to account for Neutron Primary Particles
    * The FTFP_BERT_HP is the high energy physics list with Neutron Ekin < 20 MeV.
    * Fritiof (FTF) uses above 3 GeV
    * BERT (Bertini-like) uses below 6 GeV
    * QGSP_BERT_HP or QGSP_BIC_HP  
    * Quark Glueon String Model (QGS) model >~ 20 GeV
    * Fritiof (FTF) model >~ 10 GeV
    * Binary Cascade Model (BIC) 
    * Bertini Cascade Model (BERT)
    * High precision Neutron Model (HP)
    * 
    * Use one of the _HP physics lists for neutrons:
    * FTFP_BERT_HP , QGSP_BERT_HP , QGSP_BIC_(All)HP , Shielding
    */
   G4int verbose_level = 1;
   const G4String physics_listName = "Shielding";
   G4PhysListFactory physics_listFactory;
   G4VModularPhysicsList* physics_list = physics_listFactory.GetReferencePhysList( physics_listName );
   physics_list->RegisterPhysics( new G4HadronPhysicsFTFP_BERT_HP(verbose_level) );
   run_manager->SetUserInitialization (physics_list);

   NUCE427LAB03::CLab03ActionInitialization* action_initialization = new NUCE427LAB03::CLab03ActionInitialization( detector_construction );
   run_manager->SetUserInitialization ( action_initialization );

   /*
    * For debugging only ...
    * run_manager->Initialize();
    * G4UImanager * ui_managerTemp = G4UImanager::GetUIpointer();
    * ui_managerTemp->ApplyCommand("/tracking/verbose 1");
    * run_manager->BeamOn(1);
     * detector_construction->set_targetThickness( 2.0 * CLHEP::cm);
    * run_manager->Initialize();
    * run_manager->BeamOn(1);
    * return 0;  
    */

   /*
    * Set up visualization and initialize.
    */
   G4VisManager * visual_manager = new G4VisExecutive();
   visual_manager->Initialize();

   /*
    * 1. Initialize the run manager.
    * 2. Simualte many events using BeamOn method.
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

   delete detector_construction;
   detector_construction = nullptr;
   return 0;
}