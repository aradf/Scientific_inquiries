#include "RadioactiveDecayDetectorConstruncion.hh"

CRadioactiveDecayDetectorConstruction::CRadioactiveDecayDetectorConstruction()
{
  fworld_size = 20*mm;
}

CRadioactiveDecayDetectorConstruction::~CRadioactiveDecayDetectorConstruction()
{
    
}


G4VPhysicalVolume * CRadioactiveDecayDetectorConstruction::Construct()
{
  /** 
    Define a material
  */   
  G4Material* air =  G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR"); 
  
  /** 
    Define World
  */   
  solid_world = new G4Box("solid_world",       /* name */
                                    fworld_size/2,
                                    fworld_size/2,
                                    fworld_size/2);     /* its size */
                   
  logic_world = new G4LogicalVolume(solid_world,             /* its solid */
                                                       air,                     /* its material */
                                                       "logic_World");          /* its name */

  physical_world = new G4PVPlacement(0,                      /* no rotation*/
                                                     G4ThreeVector(),            /* at (0,0,0) */
                                                     logic_world,                 /* its logical volume*/
                                                     "physical_world",           /* its name */
                                                     0,                          /* its mother  volume */
                                                     false,                      /* no boolean operation */
                                                     0);                         /* copy number */
                
  /** 
    always return the physical World
  */   
  return physical_world;

}




