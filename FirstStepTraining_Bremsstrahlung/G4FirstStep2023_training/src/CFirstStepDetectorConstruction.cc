#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4RunManager.hh"

#include "CFirstStepDetectorConstruction.hh"
namespace FS
{

CFirstStepDetectorConstruction::CFirstStepDetectorConstruction() 
: G4VUserDetectorConstruction() 
{
   ftarget_material = nullptr;
   ftarget_physicalVolume = nullptr;
   // ftarget_thickness = 1.0 * CLHEP::cm;
   ftarget_thickness = 7.0 * CLHEP::um;
   fgun_xposition = 0.0 ;
   // fgun_xposition = -0.525 * CLHEP::cm;
   this->set_targetMaterial("G4_W");
}

CFirstStepDetectorConstruction::~CFirstStepDetectorConstruction()
{

}

void CFirstStepDetectorConstruction::set_targetMaterial(const G4String& nist_material)
{
   G4NistManager *nist_manager = G4NistManager::Instance();
   G4Material * new_material = nist_manager->FindOrBuildMaterial(nist_material);
   G4cout << new_material << G4endl;
   if (new_material == nullptr)
   {
     G4cerr << "Error: CFirstStepDetectorConstruction::set_targetMaterial \n"
            << "material = "
            << new_material
            << "is not in DB ... "
            << G4endl;
     exit(-1);
   }
   if (this->ftarget_material != new_material)
   {
     this->ftarget_material = new_material;
     G4RunManager::GetRunManager()->PhysicsHasBeenModified();
   }
}

void CFirstStepDetectorConstruction::set_targetThickness(const G4double target_thckness)
{
  this->ftarget_thickness = target_thckness;
  G4RunManager::GetRunManager()->ReinitializeGeometry();
}

G4VPhysicalVolume* CFirstStepDetectorConstruction::Construct()
{
   G4cout << "Info: CFirstStepDetectorConstruction::Construct ... " << G4endl;

   /* 
    * Option to switch on/off checking of volumes overlaps
    */ 
   G4bool checkOverlaps = true;

   /*
    * 1. Create/get World Material 
    *    Get nist material manager
    */
   G4NistManager* nist = G4NistManager::Instance();
   G4Material* material_world = nist->FindOrBuildMaterial("G4_Galactic");
   
   /*
    * 2. Create World Geometry
    *    Already have target thickness 
    *    Define world and target size.
    */
   G4double target_xsize  = ftarget_thickness;
   G4double target_yzsize = 1.25 * target_xsize;
   // G4double world_xsize   = 1.1 * target_xsize;
   // G4double world_yzsize  = 1.1 * target_yzsize;
   G4double world_xsize   = 1.5 * target_xsize;
   G4double world_yzsize  = 1.5 * target_yzsize;

   /*
    * Set proper gun-x position 
    */
   fgun_xposition = -0.25 * ( target_xsize + world_xsize);

   /*
    * 3. Create World Box at (0, 0, 0)
    *    Solid is created in shape of a Box, with half-length in x, y and z.
    */
   G4Box * world_solid = new G4Box("world-solid", 
                                   0.5 * world_xsize, 
                                   0.5 * world_yzsize, 
                                   0.5 * world_yzsize);  

   /*
      Logical volume is created.  Must have pointer to solid, pointer to material, pointer to name
      and optional fields.  Optional fields are Field Manager, Sensetive Detector, user limits
      and a boolean value.
   */
   G4LogicalVolume * world_logical = new G4LogicalVolume(world_solid,  
                                                       material_world, 
                                                       "logic-World");  


   /*
      Physical volume is created.  It is a positioned instance of the logicla volume inside 
      another logical volume like the mother volume.  Could be rotated or translated relative 
      to the coordinate system of the mother volume..  
   */
   G4VPhysicalVolume * world_phyiscal = new G4PVPlacement(nullptr,             // no rotation
                                                          G4ThreeVector(0.0, 0.0, 0.0),     // at (0,0,0)
                                                          world_logical,          // its logical volume
                                                          "World",              // its name
                                                          nullptr,              // its mother  volume
                                                          false,                // no boolean operation
                                                          0);                   // copy number

   /*
    * 3. Create target Box at (0, 0, 0) inside of the world.
    *    Solid is created in shape of a Box, with half-length in x, y and z.
    */
   G4Box * target_solid = new G4Box("target-solid", 
                                     0.5 * target_xsize, 
                                     0.5 * target_yzsize, 
                                     0.5 * target_yzsize);  

   G4LogicalVolume * target_logical = new G4LogicalVolume(target_solid,  
                                                        ftarget_material, 
                                                        "logic-Target");  

   ftarget_physicalVolume = new G4PVPlacement(nullptr,                      // no rotation
                                              G4ThreeVector(0.0, 0.0, 0.0), // at (0,0,0)
                                              target_logical,               // its logical volume
                                              "Target",                     // its name
                                              world_logical,                // its mother logical volume
                                              false,                        // no boolean operation
                                              0);                           // copy number

   /* 
    * 4. Always return the world physical volume
    */
   return world_phyiscal;
}

}
