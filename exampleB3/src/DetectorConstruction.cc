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
#include "DetectorConstruction.hh"

#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4RotationMatrix.hh"
#include "G4Transform3D.hh"
#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4PSDoseDeposit.hh"
#include "G4VisAttributes.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

namespace B3
{

DetectorConstruction::DetectorConstruction()
{
  DefineMaterials();
}

void DetectorConstruction::DefineMaterials()
{
  // Retrieve elements and materials from Nist manager.
  G4NistManager* man = G4NistManager::Instance();
  G4bool isotopes = false;
  G4Element*  O = man->FindOrBuildElement("O" , 
                                           isotopes);
  G4Element* Si = man->FindOrBuildElement("Si", 
                                           isotopes);
  G4Element* Lu = man->FindOrBuildElement("Lu", 
                                           isotopes);
  // Describe the macroscopic properties of the matter like Temperature,
  // pressure, state, density, radiation length, absorption length, etc...
  auto LSO = new G4Material("Lu2SiO5", 
                             7.4 * g / cm3, 
                             3);
  LSO->AddElement(Lu, 2);
  LSO->AddElement(Si, 1);
  LSO->AddElement(O , 5);
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  // Gamma detector Parameters
  G4double cryst_dX = 6*cm, cryst_dY = 6*cm, cryst_dZ = 3*cm;
  G4int nb_cryst = 32;
  G4int nb_rings = 9;

  G4double dPhi = twopi/nb_cryst, half_dPhi = 0.5*dPhi;
  G4double cosdPhi = std::cos(half_dPhi);
  G4double tandPhi = std::tan(half_dPhi);

  G4double ring_R1 = 0.5*cryst_dY/tandPhi;
  G4double ring_R2 = (ring_R1+cryst_dZ)/cosdPhi;

  G4double detector_dZ = nb_rings*cryst_dX;

  G4NistManager* nist = G4NistManager::Instance();
  G4Material* default_mat = nist->FindOrBuildMaterial("G4_AIR");
  G4Material* cryst_mat   = nist->FindOrBuildMaterial("Lu2SiO5");

  // World
  G4double world_sizeXY = 2.4*ring_R2;
  G4double world_sizeZ  = 1.2*detector_dZ;

  // Define detector geometry
  auto solidWorld = new G4Box("World",  // its name
                              0.5 * world_sizeXY, 
                              0.5 * world_sizeXY, 
                              0.5 * world_sizeZ);  // its size

  // Logical volume = material, sensitivity and Solid
  auto logicWorld = new G4LogicalVolume(solidWorld,   // its solid
                                        default_mat,  // its material
                                        "World");     // its name

  auto physWorld = new G4PVPlacement(nullptr,           // no rotation
                                     G4ThreeVector(),   // at (0,0,0)
                                     logicWorld,        // its logical volume
                                     "World",           // its name
                                     nullptr,           // its mother  volume
                                     false,             // no boolean operation
                                     0,                 // copy number
                                     fCheckOverlaps);   // checking overlaps

  // Define a Ring to be a solid needed for Geometry.
  auto solidRing = new G4Tubs("Ring", 
                               ring_R1, 
                               ring_R2, 
                               0.5 * cryst_dX, 
                               0., 
                               twopi);

  // Logical volume = material, sensitivity and Solid
  auto logicRing = new G4LogicalVolume(solidRing,    // its solid
                                       default_mat,  // its material
                                       "Ring");      // its name

  // Define crystal to be a solid needed for Geometry.
  G4double gap = 0.5*mm;        //a gap for wrapping
  G4double dX = cryst_dX - gap, dY = cryst_dY - gap;
  auto solidCryst = new G4Box("crystal", 
                               dX / 2, 
                               dY / 2, 
                               cryst_dZ / 2);

  // Logical volume = material, sensitivity and Solid
  auto logicCryst = new G4LogicalVolume(solidCryst,     // its solid
                                        cryst_mat,      // its material
                                        "crystal_logicalVolume");   // its name

  // place crystals within a ring
  for (G4int icrys = 0; icrys < nb_cryst ; icrys++) 
  {
    G4double phi = icrys*dPhi;
    G4RotationMatrix rotm  = G4RotationMatrix();
    rotm.rotateY(90*deg);
    rotm.rotateZ(phi);
    G4ThreeVector uz = G4ThreeVector(std::cos(phi),  
                                     std::sin(phi),
                                     0.);

    G4ThreeVector position = (ring_R1+0.5*cryst_dZ)*uz;
    G4Transform3D transform = G4Transform3D(rotm,
                                            position);

    // A physical volume is positioned instance of a logical volume inide
    // another logical volume (World or Mother).
    new G4PVPlacement(transform,             //rotation,position
                      logicCryst,            //its logical volume
                      "crystal",             //its name
                      logicRing,             //its mother  volume
                      false,                 //no boolean operation
                      icrys,                 //copy number
                      fCheckOverlaps);       // checking overlaps
  }

  // Define the full detector to be a Solid needed for Geometry.
  auto solidDetector = new G4Tubs("Detector", 
                                   ring_R1, 
                                   ring_R2, 
                                   0.5 * detector_dZ, 
                                   0., 
                                   twopi);

  auto logicDetector = new G4LogicalVolume(solidDetector,  // its solid
                                           default_mat,    // its material
                                           "Detector");    // its name

  // Place physical volume of rings within detector logical volume (World or Mother).
  G4double OG = -0.5*(detector_dZ + cryst_dX);
  for (G4int iring = 0; iring < nb_rings ; iring++) 
  {
      OG += cryst_dX;
      new G4PVPlacement(nullptr,                  // no rotation
                        G4ThreeVector(0, 0, OG),  // position
                        logicRing,                // its logical volume
                        "ring",                   // its name
                        logicDetector,            // its mother  volume
                        false,                    // no boolean operation
                        iring,                    // copy number
                        fCheckOverlaps);          // checking overlaps
  }

  // Place detector physical volume in world physical volume
  new G4PVPlacement(nullptr,                  // no rotation
                    G4ThreeVector(),          // at (0,0,0)
                    logicDetector,            // its logical volume
                    "Detector",               // its name
                    logicWorld,               // its mother  volume
                    false,                    // no boolean operation
                    0,                        // copy number
                    fCheckOverlaps);          // checking overlaps

  // patient
  G4double patient_radius = 8*cm;
  G4double patient_dZ = 10*cm;
  G4Material* patient_mat = nist->FindOrBuildMaterial("G4_BRAIN_ICRP");

  // Define a solid needed for the geometry.
  auto solidPatient = new G4Tubs("patient", 
                                  0., 
                                  patient_radius, 
                                  0.5 * patient_dZ, 
                                  0., 
                                  twopi);

  // Logical volume = material, sensitivity and Solid
  auto logicPatient = new G4LogicalVolume(solidPatient,     // its solid
                                          patient_mat,      // its material
                                          "patient_logicalvolume");     // its name

  // Place patient logical volume in world volume
  new G4PVPlacement(nullptr,                  // no rotation
                    G4ThreeVector(),          // at (0,0,0)
                    logicPatient,             // its logical volume
                    "Patient",                // its name
                    logicWorld,               // its mother  volume
                    false,                    // no boolean operation
                    0,                        // copy number
                    fCheckOverlaps);          // checking overlaps

  // Visualization attributes
  logicRing->SetVisAttributes (G4VisAttributes::GetInvisible());
  logicDetector->SetVisAttributes (G4VisAttributes::GetInvisible());

  // Print materials
  G4cout << *(G4Material::GetMaterialTable()) << G4endl;

  //always return the physical World
  return physWorld;
}

// Construct the Sensitive Ditector Field.
void DetectorConstruction::ConstructSDandField()
{
  G4SDManager::GetSDMpointer()->SetVerboseLevel(1);

  // Declare crystal as a MultiFunctionalDetector scorer object.
  // It creates a collections (maps) for the cystal scorer
  // G4's Sensitive Detector Manager Pointer.
  // Register Cryst instance of G4MultiFunctionalDetector class type
  // to the primtiv1 instance of the G4VPrimitiveScorer class type.
  // Store the sum of score for a particle's energy deposites at 
  // each step in the cell.  
  auto cryst = new G4MultiFunctionalDetector("crystal");
  G4SDManager::GetSDMpointer()->AddNewDetector(cryst);
  G4VPrimitiveScorer* primitive_energydeposite = new G4PSEnergyDeposit("edep");
  cryst->RegisterPrimitive(primitive_energydeposite);
  SetSensitiveDetector("crystal_logicalVolume",
                        cryst);

  // Declare patient as a MultiFunctionalDetector scorer object
  auto patient = new G4MultiFunctionalDetector("patient");
  G4SDManager::GetSDMpointer()->AddNewDetector(patient);
  G4VPrimitiveScorer* primitive_dosedeposite = new G4PSDoseDeposit("dose");
  patient->RegisterPrimitive(primitive_dosedeposite);
  SetSensitiveDetector("patient_logicalvolume",
                        patient);
}

}

