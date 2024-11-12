#include <G4VPhysicalVolume.hh>
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Element.hh"
#include "G4SystemOfUnits.hh"
#include <G4VisAttributes.hh>
#include <G4RunManagerFactory.hh>

#include "CLab03DetectorConstruction.hh"

namespace NUCE427LAB03
{

CLab03DetectorConstructor::CLab03DetectorConstructor() : G4VUserDetectorConstruction()
{
   G4cout << "INFO: CLab03DetectorConstructor Constructor ..."
          << G4endl;

   ftarget_physicalVolume = nullptr;
   fmoderator_material = nullptr;
   fshield_material = nullptr;
   ftarget_material = nullptr;
   fgun_xposition = 0.0;

   ftarget_thickness = 1.0 * CLHEP::cm;

   // TBD:  The fgun_xposition should be removed from here
   fgun_xposition = ftarget_thickness * 100 *(-0.9);
   set_moderatorMaterial();
   set_shieldMaterial();
   set_targetMaterial();
   G4cout << "INFO: Materials"
          << G4endl
          << "INFO: Moderator "
          << this->fmoderator_material
          << G4endl
          << "INFO: Shiled "
          << this->fshield_material
          << G4endl
          << "INFO: Target"
          << this->ftarget_material
          << G4endl;
}

CLab03DetectorConstructor::~CLab03DetectorConstructor()
{
   G4cout << "INFO: CLab03DetectorConstructor Destructor ..."
          << G4endl;
}

void CLab03DetectorConstructor::set_moderatorMaterial() 
{
   /*
    * Paraffin == C20-H42
    */
   G4NistManager* nist_manager = G4NistManager::Instance();
   G4bool isotopes = false;
   G4int natoms;
   G4Element* element_hydrogen = nist_manager->FindOrBuildElement("H" , isotopes);
   G4Element* element_carbon   = nist_manager->FindOrBuildElement("C" , isotopes);

   if (element_hydrogen == nullptr)
   {
      G4cout << "INFO: Material is incorrect ..." 
             << element_hydrogen;
      return;
   }

   if (element_carbon == nullptr)
   {
      G4cout << "INFO: Material is incorrect ..." 
             << element_carbon;
      return;             
   }

   fmoderator_material = new G4Material( "Paraffin", 
                                         0.7886  * g/cm3,  
                                         2  );

   fmoderator_material->AddElement( element_hydrogen, 
                                    natoms = 42 );
   fmoderator_material->AddElement( element_carbon, 
                                    natoms = 20 );

}

void CLab03DetectorConstructor::set_targetMaterial() 
{
   /*
    * Detector == BF3
    */
   G4NistManager* nist_manager = G4NistManager::Instance();
   G4bool isotopes = false;
   G4int natoms;
   G4Element* element_boron = nist_manager->FindOrBuildElement("B" , isotopes);
   G4Element* element_flourine   = nist_manager->FindOrBuildElement("F" , isotopes);

   if (element_boron == nullptr)
   {
      G4cout << "INFO: Material is incorrect ..." 
             << element_boron;
      return;             
   }

   if (element_flourine == nullptr)
   {
      G4cout << "INFO: Material is incorrect ..." 
             << element_flourine;
      return;             
   }

   if (ftarget_material == nullptr)
   {
      ftarget_material = new G4Material( "Detector", 
                                          1.6  * g/cm3,  
                                          2  );

      ftarget_material->AddElement( element_boron, 
                                       natoms = 1 );
      ftarget_material->AddElement( element_flourine, 
                                       natoms = 3 );
      
      G4RunManager::GetRunManager()->PhysicsHasBeenModified();
   }

}

void CLab03DetectorConstructor::set_targetThickness(const G4double target_thickness)
{
   ftarget_thickness = target_thickness;
   G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void CLab03DetectorConstructor::set_shieldMaterial() 
{
   /*
    * shield is pb
    */
   G4NistManager* nist_manager = G4NistManager::Instance();
   fshield_material = nist_manager->FindOrBuildMaterial("G4_Pb");

   if (fshield_material == nullptr)
   {
      G4cout << "INFO: Material is incorrect ..." 
             << fshield_material;
      return;             
   }
}

/*
 * Material definition
 *    - NIST material data base; 
 * Geometry definition
 *    - solid, logical volume, physical volume.
 */
G4VPhysicalVolume* CLab03DetectorConstructor::Construct()
{
   G4cout << "INFO: CLab03DetectorConstructor::Constructor ... " 
          << G4endl;    
   /*
    * 1.a Create/get World Material 
    *    Get nist material manager
    */
   G4NistManager* nist_manager = G4NistManager::Instance();
   G4Material* world_material = nist_manager->FindOrBuildMaterial("G4_AIR");
   G4Material* moderator_material = get_moderatorMaterial();
   G4Material* target_material = get_targetMaterial();
   const G4Material* shield_material = get_shieldMaterial();

   /*
    * 1.b Create World Geometry
    *    Already have target thickness 
    *    Define world and target size.
    */
   // ftarget_thickness = target_xsize = 1.0 cm;
   // target_yzsize     = 10.0 cm;
   // world_xsize       = 100.0 cm;
   // world_yzsize      = 50.0 cm;
   G4double target_xsize   = ftarget_thickness;
   // G4double target_yzsize  = 10  * target_xsize;
   G4double target_yzsize  = 30  * target_xsize;

   G4double world_xsize    = 100 * target_xsize;
   G4double world_yzsize   = 50  * target_xsize;

   // TBD: Add the fgun_xposition back for production ...
   // fgun_xposition = target_xsize * (-1.1);

   /*
    * 1.c Create World Box at (0, 0, 0)
    *    Solid is created in shape of a Box, with half-length in x, y and z.
    */
   G4Box * world_solid = new G4Box( "world-solid", 
                                    0.5 * world_xsize, 
                                    0.5 * world_yzsize, 
                                    0.5 * world_yzsize );  

  /*
      Logical volume is created.  Must have pointer to solid, pointer to material, pointer to name
      and optional fields.  Optional fields are Field Manager, Sensetive Detector, user limits
      and a boolean value.
   */
   G4LogicalVolume * world_logical = new G4LogicalVolume( world_solid,  
                                                          world_material, 
                                                          "logic-World");  

   /* Add color to the world. */
   G4VisAttributes* world_visualAttribute = new G4VisAttributes();
   world_visualAttribute->SetVisibility(false);
   world_logical->SetVisAttributes(world_visualAttribute);

   /*
      Physical volume is created.  It is a positioned instance of the logicla volume inside 
      another logical volume like the mother volume.  Could be rotated or translated relative 
      to the coordinate system of the mother volume..  
   */
   G4VPhysicalVolume* world_physicalVolume = new G4PVPlacement( nullptr,                        // no rotation
                                                                G4ThreeVector(0.0, 0.0, 0.0),   // at (0,0,0)
                                                                world_logical,                  // its logical volume
                                                                "World",                        // its name
                                                                nullptr,                        // its mother  volume
                                                                false,                          // no boolean operation
                                                                0 );                            // copy number

   /*
    * 2. Create Moderator Box at (0, 0, 0) inside of the world.
    *    Solid is created in shape of a Box, with half-length in x, y and z.
    */
   G4double moderator_xsize    = 0.150 * world_xsize;
   G4double moderator_yzsize   = 0.150 * world_yzsize;

   // moderator_xsize   = 50.0 cm;
   // moderator_yzsize  = 25.0 cm;
   G4Box* moderator_solid = new G4Box( "moderator-solid", 
                                       0.5 * moderator_xsize, 
                                       0.5 * moderator_yzsize, 
                                       0.5 * moderator_yzsize);  

   G4LogicalVolume* moderator_logical = new G4LogicalVolume( moderator_solid,  
                                                             moderator_material, 
                                                             "logic-Moderator");  
   
   /* Colorize the moderator using the blue visualization attributes. */   
   G4VisAttributes* moderator_blue = new G4VisAttributes(G4Colour::Blue());
   moderator_blue->SetVisibility(true);
   moderator_logical->SetVisAttributes(moderator_blue);

   G4VPhysicalVolume * moderator_physicalVolume = new G4PVPlacement( nullptr,                      // no rotation
                                                                     G4ThreeVector(0.0, 0.0, 0.0), // at (0,0,0)
                                                                     moderator_logical,            // its logical volume
                                                                     "moderator",                  // its name
                                                                     world_logical,                // its mother logical volume
                                                                     false,                        // no boolean operation
                                                                     0);                           // copy number

   /*
    * Create target definition.
    */
   G4Box * target_solid = new G4Box( "target-solid", 
                                      0.5 * target_xsize, 
                                      0.5 * target_yzsize, 
                                      0.5 * target_yzsize );  

   G4LogicalVolume* target_logical = new G4LogicalVolume( target_solid,  
                                                          target_material, 
                                                          "logic-Target");  

   ftarget_physicalVolume = new G4PVPlacement( nullptr,                      // no rotation
                                               G4ThreeVector(( moderator_xsize + 100 ) / 2.0 , 0.0, 0.0), // at ( (100 - 200) / 2.0, 0, 0)
                                               target_logical,            // its logical volume
                                               "target",                  // its name
                                               world_logical,                // its mother logical volume
                                               false,                        // no boolean operation
                                               0);                           // copy number

   /* Colorize the target using the blue visualization attributes. */   
   G4VisAttributes* target_red = new G4VisAttributes(G4Colour::Red());
   target_red->SetVisibility(true);
   target_logical->SetVisAttributes(target_red);

   return world_physicalVolume;
}

}// end of name space 