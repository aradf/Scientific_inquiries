#include "RadioactiveDecayDetectorConstruncion.hh"

CRadioactiveDecayDetectorConstruction::CRadioactiveDecayDetectorConstruction()
{
   define_material();

   xworld = 0.5*m;
   yworld = 0.5*m;
   zworld = 0.5*m;
}

CRadioactiveDecayDetectorConstruction::~CRadioactiveDecayDetectorConstruction()
{
    
}

void CRadioactiveDecayDetectorConstruction::define_material()
{
      /**
    * Get an instance of the Nist Manager for the world volume.
    */
   G4NistManager  * nist_manager = G4NistManager::Instance();
   world_material = nist_manager->FindOrBuildMaterial("G4_AIR");

   G4double energy[2] = {1.239841939*eV/0.9, 1.239841939*eV/0.2};
   G4double rindexWorld[2] = {1.0, 1.0};

   G4MaterialPropertiesTable *mptWorld = new G4MaterialPropertiesTable();
   mptWorld->AddProperty("RINDEX", 
                          energy, 
                          rindexWorld, 
                          2);

   world_material->SetMaterialPropertiesTable(mptWorld);
   Na = nist_manager->FindOrBuildElement("Na");
   I = nist_manager->FindOrBuildElement("I");
   NaI = new G4Material("NaI", 3.67*g/cm3, 2);
   NaI->AddElement(Na, 1);
   NaI->AddElement(I, 1);

   /*
   mirror_surface = new G4OpticalSurface("mirror_surface");
   mirror_surface->SetType(dielectric_metal);
   mirror_surface->SetFinish(ground);
   mirror_surface->SetModel(unified);

   G4double rindexNaI[2] = {1.78, 1.78};
   G4double reflectivity[2] = {1.0, 1.0};
   G4double fraction[2] = {1.0, 1.0};
   G4MaterialPropertiesTable *mptNaI = new G4MaterialPropertiesTable();
   mptNaI->AddProperty("RINDEX", energy, rindexNaI, 2);
   mptNaI->AddProperty("SCINTILLATIONCOMPONENT1", energy, fraction, 2);
   mptNaI->AddConstProperty("SCINTILLATIONYIELD", 38.0/keV);
   mptNaI->AddConstProperty("RESOLUTIONSCALE", 1.0);
   mptNaI->AddConstProperty("SCINTILLATIONTIMECONSTANT1", 250*ns);
   mptNaI->AddConstProperty("SCINTILLATIONYIELD1", 1.0);

   NaI->SetMaterialPropertiesTable(mptNaI);
   
   G4MaterialPropertiesTable * mptMirror = new G4MaterialPropertiesTable();
   mptMirror->AddProperty("REFLECTIVITY", energy, reflectivity, 2);

   mirror_surface->SetMaterialPropertiesTable(mptMirror);
*/

}

void CRadioactiveDecayDetectorConstruction::construct_scintillator()
{
   scintillator_solid = new G4Box("scintillator_solid", 
                                   5*cm, 
                                   5*cm, 
                                   6*cm);

   scintillator_logic = new G4LogicalVolume(scintillator_solid, 
                                            NaI, 
                                            "scintillator_logic");

   G4LogicalSkinSurface * skin = new G4LogicalSkinSurface("skin", 
                                                           world_logic, 
                                                           mirror_surface);

   detector_solid = new G4Box("detector_solid", 
                              1*cm, 
                              5*cm, 
                              6*cm);

   detector_logic = new G4LogicalVolume(detector_solid, 
                                        world_material, 
                                        "detector_logic");


   fscoring_volume = scintillator_logic;

  for (G4int i = 0 ; i < 6; i++)
   {
      for (G4int j = 0 ; j < 16; j++)
      {
         G4Rotate3D rotateZ(j*22.5*deg, G4ThreeVector(0,0,1));
         G4Translate3D transXScint(G4ThreeVector (5./tan(22.5/2*deg)*cm + 5.0*cm, 
                                                  0*cm, 
                                                  -40*cm + i*15*cm));

         G4Translate3D transXDet(G4ThreeVector (5./tan(22.5/2*deg)*cm + 6.0*cm + 5.0*cm, 
                                                0*cm, 
                                                -40*cm + i*15*cm));

         G4Transform3D transform_scintillator = (rotateZ)*(transXScint);
         G4Transform3D transform_detector = (rotateZ)*(transXDet);

         scintillator_physics = new G4PVPlacement(transform_scintillator,
                                                   scintillator_logic,
                                                   "scintillator_physics", 
                                                   world_logic, 
                                                   false, 
                                                   0, 
                                                   true);                       

         detector_physical = new G4PVPlacement(transform_detector,
                                                   detector_logic,
                                                   "detector_physical", 
                                                   world_logic, 
                                                   false, 
                                                   0, 
                                                   true);                       

      }
   }
   
   scintillator_physics = new G4PVPlacement(0, 
                                            G4ThreeVector(0.0, 0.0, 0.0),
                                            scintillator_logic,
                                            "scintillator_physics", world_logic, false, 0, true);                       

}


G4VPhysicalVolume * CRadioactiveDecayDetectorConstruction::Construct()
{
   G4bool check_overlaps = true;

   /**
    * The solid defines the boundaries and size of the wolid.
    * In this case is 1.0 x 1.0 x 1.0 [m^3]
    */
   world_solid = new G4Box("world_solid", 
                            xworld, 
                            yworld, 
                            zworld);  

   /**
    * The logical volume will include the material.
    */
   world_logic = new G4LogicalVolume(world_solid, 
                                     world_material, 
                                     "world_logic");

   world_physics = new G4PVPlacement(0,  
                                     G4ThreeVector(0, 0, 0),  
                                     world_logic, 
                                     "world_physics", 
                                     0,  
                                     false,   
                                     0,  
                                     check_overlaps);

   /**
    * Add the photon or sensetive detector.
    */
    construct_scintillator();

   return world_physics;
}

void CRadioactiveDecayDetectorConstruction::ConstructSDandField()
{
   CRadioactiveDecaySensitiveDetector * sensitive_detector = new CRadioactiveDecaySensitiveDetector("Sensitive_detector");
   if (detector_logic != NULL)
   {
      detector_logic->SetSensitiveDetector(sensitive_detector);
   }

}



