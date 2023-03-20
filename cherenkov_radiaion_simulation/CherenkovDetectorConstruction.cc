#include "CherenkovDetectorConstruction.hh"

/**
 * Constructor Description.
 * 
 */
CCherenkovDetectorConstruction::CCherenkovDetectorConstruction()
{
   /**
    * Define UI commands usigng the G4GenericMessenger object and DeclareProperty.
    */
   fmessenger = new G4GenericMessenger(this, "/cherenkov_detector/", "Detector Construction");
   fmessenger->DeclareProperty("number_columns", number_columns, "Number of columns");
   fmessenger->DeclareProperty("number_rows", number_rows, "Number of rows");

   number_columns = 100;
   number_rows = 100;

   xworld = 0.5*m;
   yworld = 0.5*m;
   zworld = 0.5*m;
 
   DefineMaterial();
}

CCherenkovDetectorConstruction::~CCherenkovDetectorConstruction()
{



}

/**
 * The solid world, logic world, physics world is defined here.
 * The solid radiator, logic radiator, physics radiator is defined here.
 * The solid photon detector, logic photon detector, physics photon detector is defined here.
 */
void CCherenkovDetectorConstruction::DefineMaterial()
{
   /**
    * Get an instance of the Nist Manager for the world,  radiator and detector.
    */
   G4NistManager  * nist_manager = G4NistManager::Instance();
   SiO2 = new G4Material("SiO2", 2.201*g/cm3, 2);
   SiO2->AddElement(nist_manager->FindOrBuildElement("Si"), 1);
   SiO2->AddElement(nist_manager->FindOrBuildElement("O"), 2);

   H2O = new G4Material("H2O", 1.000*g/cm3, 2);
   H2O->AddElement(nist_manager->FindOrBuildElement("H"), 2);
   H2O->AddElement(nist_manager->FindOrBuildElement("O"), 1);
   C = nist_manager->FindOrBuildElement("C");

   Aerogel = new G4Material("Aerogel", 0.200*g/cm3, 3);
   Aerogel->AddMaterial(SiO2, 62.5*perCent);
   Aerogel->AddMaterial(H2O, 37.4*perCent);
   Aerogel->AddElement(C, 0.1*perCent);

   // This next line works in an older version of geant4
   // G4double energy[2] = {1.239841939*eV/0.2, 1.239841939*eV/0.9}; 
   G4double energy[2] = {1.239841939*eV/0.9, 1.239841939*eV/0.2};
   G4double rindexAerogel[2] = {1.1, 1.1};
   
   //using the refraction index for water.
   //G4double rindexAerogel[2] = {1.3, 1.3};
   G4double rindexWorld[2] = {1.0, 1.0};

   world_material = nist_manager->FindOrBuildMaterial("G4_AIR");

   G4MaterialPropertiesTable *mptAerogel = new G4MaterialPropertiesTable();
   mptAerogel->AddProperty("RINDEX", 
                           energy, 
                           rindexAerogel, 
                           2);

   G4MaterialPropertiesTable *mptWorld = new G4MaterialPropertiesTable();
   mptWorld->AddProperty("RINDEX", 
                          energy, 
                          rindexWorld, 
                          2);

   Aerogel->SetMaterialPropertiesTable(mptAerogel);


   world_material->SetMaterialPropertiesTable(mptWorld);


}

/**
 * The solid world, logic world, physics world is defined here.
 * The solid radiator, logic radiator, physics radiator is defined here.
 * The solid photon detector, logic photon detector, physics photon detector is defined here.
 */
G4VPhysicalVolume * CCherenkovDetectorConstruction::Construct()
{

   G4bool check_overlaps = true;

   /**
    * Get an instance of the Nist Manager for the world volume.
    */
   G4NistManager  * nist_manager = G4NistManager::Instance();
   
     

   /**
    * The solid defines the boundaries and size of the wolid.
    * In this case is 1.0 x 1.0 x 1.0 [m^3]
    */
   solid_world = new G4Box("solid_world", 
                            xworld, 
                            yworld, 
                            zworld);  

   /**
    * The logical volume will include the material.
    */
   logic_world = new G4LogicalVolume(solid_world, 
                                     world_material, 
                                     "logic_world");

   physics_world = new G4PVPlacement(0,  
                                     G4ThreeVector(0, 0, 0),  
                                     logic_world, 
                                     "physics_World", 
                                     0,  
                                     false,   
                                     0,  
                                     check_overlaps);
 
 
   /////////////////////////////////////
   solid_radiator = new G4Box("solid_radiator", 
                               0.4*m, 
                               0.4*m, 
                               0.01*m);

   logic_radiator = new G4LogicalVolume(solid_radiator, 
                                        Aerogel, 
                                        "logic_radiator");

   fscoring_volume = logic_radiator;
   
   physics_radiator = new G4PVPlacement(0,  
                                        G4ThreeVector(0.0, 0.0, 0.25*m),  
                                         logic_radiator, 
                                        "physics_radiator", 
                                        logic_world,  
                                        false,   
                                        0,  
                                        check_overlaps);
   
   ///////////////////////////////////
   solid_detector = new G4Box("solid_detector", 
                               xworld/number_rows, 
                               yworld/number_columns, 
                               0.01*m);
                               
   logic_detector = new G4LogicalVolume(solid_detector, 
                                        world_material, 
                                        "logic_detector");

   for (G4int i = 0 ; i < number_rows ; i++)
   {
       for (G4int j = 0 ; j < number_columns; j++)
       {
           physics_detector = new G4PVPlacement(0, 
                                                 G4ThreeVector(-0.5*m + (i+0.5)*m/number_rows, -0.5*m+(j+0.5)*m/number_columns , 0.49*m),
                                                 logic_detector, 
                                                 "physics_detector", 
                                                 logic_world,
                                                 false,
                                                 j+i*number_columns, 
                                                 check_overlaps);
       }
   }


   return physics_world;
}

/** 
 * In this method define sensitive detectors and assign them to volumes
 * Assign magnetic field to volumes / regions.
 */
void CCherenkovDetectorConstruction::ConstructSDandField()
{
   CCherenkovSensitiveDetector * sensitive_detector = new CCherenkovSensitiveDetector("Sensitive_detector");
   
   logic_detector->SetSensitiveDetector(sensitive_detector)  ;

}
