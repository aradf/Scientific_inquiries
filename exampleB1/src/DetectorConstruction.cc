//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
/// \file B1/src/DetectorConstruction.cc
/// \brief Implementation of the B1::DetectorConstruction class

#include "DetectorConstruction.hh"

#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Orb.hh"
#include "G4Sphere.hh"
#include "G4Trd.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"

namespace B1
{

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  // Get nist material manager
  G4NistManager* nist = G4NistManager::Instance();

  // Envelope parameters
  //
  G4double env_sizeXY = 20*cm, env_sizeZ = 30*cm;
  //G4Material* env_mat = nist->FindOrBuildMaterial("G4_WATER");
  G4Material* env_mat = nist->FindOrBuildMaterial("G4_AIR");

  // Option to switch on/off checking of volumes overlaps
  //
  G4bool checkOverlaps = true;

  //
  // World - All solids, logical volumes and their physical volue are placed in this mohter volume.
  //
  G4double world_sizeXY = 1.2*env_sizeXY;
  G4double world_sizeZ  = 1.2*env_sizeZ;
  G4Material* world_mat = nist->FindOrBuildMaterial("G4_AIR");

  // Solid is created in shape of a Box, with half-length in x, y and z.
  auto solidWorld = new G4Box("World", 
                               0.5 * world_sizeXY, 
                               0.5 * world_sizeXY, 
                               0.5 * world_sizeZ);  

  // Logical volume is created.  Must have pointer to solid, pointer to material, pointer to name
  // and optional fields.  Optional fields are Field Manager, Sensetive Detector, user limits
  // and a boolean value.
  auto logicWorld = new G4LogicalVolume(solidWorld,  
                                        world_mat, 
                                        "World");  

  // Physical volume is created.  It is a positioned instance of the logicla volume inside 
  // another logical volume like the mother volume.  Could be rotated or translated relative 
  // to the coordinate system of the mother volume..  
  auto physWorld = new G4PVPlacement(nullptr,             // no rotation
                                     G4ThreeVector(),     // at (0,0,0)
                                     logicWorld,          // its logical volume
                                     "World",             // its name
                                     nullptr,             // its mother  volume
                                     false,               // no boolean operation
                                     0,                   // copy number
                                     checkOverlaps);      // overlaps checking

  //
  // Envelope:  Create solid for Envelope the shape of a box with with half-length of X, Y, and Z.
  //
  auto solidEnv = new G4Box("Envelope",                    // its name
                             0.5 * env_sizeXY, 
                             0.5 * env_sizeXY, 
                             0.5 * env_sizeZ);             // its size

  //
  // logical volume envelop: Create logical volume for envelope with material of G4_water.
  //
  auto logicEnv = new G4LogicalVolume(solidEnv,           // its solid
                                      env_mat,            // its material
                                      "Envelope");        // its name

  // Physical volume: Create physical volume for the envelope.  with no rotation and translation.
  // placed in the world (mother) volume.
  new G4PVPlacement(nullptr,                  // no rotation
                    G4ThreeVector(),          // at (0,0,0)
                    logicEnv,                 // its logical volume
                    "Envelope",               // its name
                    logicWorld,               // its mother  volume
                    false,                    // no boolean operation
                    0,                        // copy number
                    checkOverlaps);           // overlaps checking

  //
  // Shape 1:  Create shape1_mat of type G4 A150 Tissue.
  //
  // G4Material* shape1_mat = nist->FindOrBuildMaterial("G4_A-150_TISSUE");
  G4Material* shape1_mat = nist->FindOrBuildMaterial("G4_BONE_COMPACT_ICRU");
  
  // Position is a vector of x = 0, y = 2 cm, and z 0f -7 cm.
  // G4ThreeVector pos1 = G4ThreeVector(0, 2*cm, -7*cm);
  G4ThreeVector pos1 = G4ThreeVector(0, 2*cm, -7*cm);

  // Conical section shape
  G4double shape1_rmina =  0.*cm, shape1_rmaxa = 2.*cm;
  G4double shape1_rminb =  0.*cm, shape1_rmaxb = 4.*cm;
  // G4double shape1_hz = 3.*cm;
  G4double shape1_hz = 0.5*cm;
  G4double shape1_phimin = 0.*deg, shape1_phimax = 360.*deg;

  // Shape 1: Create solid in shape of a cone with the follwoing parameters.
  // name
  // inner radius -pDz
  // outer radius -pDz
  // inner radius +pDz
  // outer radius +pDz
  // Z half length
  // starting Phi
  // segment angle
  auto solidShape1 = new G4Cons("Shape1", 
                                 shape1_rmina, 
                                 shape1_rmaxa, 
                                 shape1_rminb, 
                                 shape1_rmaxb,
                                 shape1_hz, 
                                 shape1_phimin, 
                                 shape1_phimax);

  // Shape 1:  Create logical Volume with the solid of type G4Cone, material A-150 TISSUE, and name 'Shape1'

  auto logicShape1 = new G4LogicalVolume(solidShape1,  // its solid
                                         shape1_mat,   // its material
                                         "Shape1");    // its name

  // Shape 1: Create a physical volume with no rotation and translation of (0, 2, -7) placed inside of the it's mother (logic env)
  new G4PVPlacement(nullptr,  // no rotation
                    pos1,                     // at position
                    logicShape1,              // its logical volume
                    "Shape1",                 // its name
                    logicEnv,                 // its mother  volume
                    false,                    // no boolean operation
                    0,                        // copy number
                    checkOverlaps);           // overlaps checking

  //
  // Shape 2: Material is defined to be Compact Bone.
  // The International Commission on Radiation Units and Measurements (ICRU)
  //
  // G4Material* shape2_mat = nist->FindOrBuildMaterial("G4_BONE_COMPACT_ICRU");
  G4Material* shape2_mat = nist->FindOrBuildMaterial("G4_A-150_TISSUE");

  // position of the shape two is translated by 0, -1 cm and +7 cm.
  G4ThreeVector pos2 = G4ThreeVector(0, -1*cm, 7*cm);

  // Trapezoid shape
  G4double shape2_dxa = 12*cm, shape2_dxb = 12*cm;
  G4double shape2_dya = 10*cm, shape2_dyb = 16*cm;
  G4double shape2_dz  = 6*cm;

  // shape2:  Define Solid to be a a trapzoid.  
  auto solidShape2 = new G4Trd("Shape2",  // its name
                                0.5 * shape2_dxa, 
                                0.5 * shape2_dxb, 
                                0.5 * shape2_dya, 
                                0.5 * shape2_dyb,
                                0.5 * shape2_dz);  // its size

  // Shape2: Define Logical volume to be have a solid of type trapzoid, material of Compact Bone.
  auto logicShape2 = new G4LogicalVolume(solidShape2,         // its solid
                                         shape2_mat,          // its material
                                         "Shape2");           // its name

  // Shape2: Define Physical volume of shape2 with no rotation, translated 0, -1, +7 cm.  It's mother volume is logic env.
  new G4PVPlacement(nullptr,                  // no rotation
                    pos2,                     // at position
                    logicShape2,              // its logical volume
                    "Shape2",                 // its name
                    logicEnv,                 // its mother  volume
                    false,                    // no boolean operation
                    0,                        // copy number
                    checkOverlaps);           // overlaps checking

  G4Material* shape3_mat = nist->FindOrBuildMaterial("G4_WATER");
  // position of the shape two is translated by 0, -1 cm and +7 cm.
  G4ThreeVector pos3 = G4ThreeVector(0.0, 0.0*cm, -1*cm);
  // Create the solid volume for the water slab    
  auto solidShape3 = new G4Box("waterslab", env_sizeXY / 3, env_sizeXY / 3, env_sizeZ / 150);
  // Shape2: Define Logical volume to be have a solid of type trapzoid, material of Compact Bone.
  auto logicShape3 = new G4LogicalVolume(solidShape3,         // its solid
                                         shape3_mat,          // its material
                                         "Shape3");           // its name
  // Shape3: Define Physical volume of shape3
  new G4PVPlacement(nullptr,                  // no rotation
                    pos3,                     // at position
                    logicShape3,              // its logical volume
                    "Shape3",                 // its name
                    logicEnv,                 // its mother  volume
                    false,                    // no boolean operation
                    0,                        // copy number
                    checkOverlaps);           // overlaps checking

  G4cout << "All Material are: " << G4endl;
  G4cout << *(G4Material::GetMaterialTable()) << G4endl;
  
  // Set Shape2 as scoring volume
  //
  fScoringVolumeTissue = logicShape2;
  fScoringVolumeBone = logicShape1;

  //
  //always return the physical World
  //
  return physWorld;
}

}
