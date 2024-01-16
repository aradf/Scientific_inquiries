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
#include "G4SDManager.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4GenericMessenger.hh"

#include "SensitiveDetector.hh"

namespace NUCE427LAB02
{

DetectorConstruction::DetectorConstruction()
{
  is_cherenkov = false;
  is_positronEmissionTomography = true;

  if (is_cherenkov) 
     material_aerogel();
  if (is_positronEmissionTomography) 
     material_NaI();

  xworld = 0.5 * m;
  yworld = 0.5 * m;
  zworld = 0.5 * m;

  fgeneric_messenger = new G4GenericMessenger(this, 
                                              "/detector/",
                                              "Detector Construction Type");
  fgeneric_messenger->DeclareProperty("number_columns", number_columns, "Number of Columns");
  fgeneric_messenger->DeclareProperty("number_rows", number_rows, "Number of Rows");

  number_rows = 100;
  number_columns = 100;

}

DetectorConstruction::~DetectorConstruction()
{
}

void DetectorConstruction::material_aerogel()
{
  G4NistManager* nist_manager = G4NistManager::Instance();
  SiO2 = new G4Material("SiO2", 2.201 * g/cm3, 2);
  SiO2->AddElement(nist_manager->FindOrBuildElement("Si"), 1);
  SiO2->AddElement(nist_manager->FindOrBuildElement("O"), 2);

  H2O = new G4Material("H2O", 1.0 * g/cm3, 2);
  H2O->AddElement(nist_manager->FindOrBuildElement("H"), 1);
  H2O->AddElement(nist_manager->FindOrBuildElement("O"), 2);

  C = nist_manager->FindOrBuildElement("C");
  aerogel = new G4Material("aerogel", 0.200 * g/cm, 3);
  aerogel->AddMaterial(SiO2, 62.5 * perCent);
  aerogel->AddMaterial(H2O, 37.4 * perCent);
  aerogel->AddElement(C, 0.1 * perCent);
  
  G4double energy[2] = {1.239841939*eV/0.9, 1.239841939*eV/0.2};
  G4double rindexAerogel[2] = {1.1, 1.1};

  G4MaterialPropertiesTable *mptAerogel = new G4MaterialPropertiesTable();
  mptAerogel->AddProperty("RINDEX", 
                           energy, 
                           rindexAerogel, 
                           2);

  aerogel->SetMaterialPropertiesTable(mptAerogel);
}

void DetectorConstruction::cherenkov_squareGeometry()
{
  solid_cherenkovsquare = new G4Box("solid_cherenkovsquare", 0.4 * m, 0.4 * m, 0.01 * m);

  logicalvolume_cherenkovsquare = new G4LogicalVolume(solid_cherenkovsquare, 
                                               aerogel, 
                                               "logicalVolumesquare");

  logicalvolume_scoring = logicalvolume_cherenkovsquare;

  physicalvolume_cherenkovsquare = new G4PVPlacement(nullptr,
                                              G4ThreeVector(0.0, 0.0, 0.25 * m),
                                              logicalvolume_cherenkovsquare,
                                              "physicalvolume_cherenkovsquare",
                                              logicalvolume_world,
                                              false,
                                              0,
                                              true);

}

void DetectorConstruction::cherenkov_sensitiveDetector()
{
   solid_cherenkovSensitiveDetector = new G4Box("solid_cherenkovSensitiveDetector",
                                      0.005*m, 
                                      0.005*m,
                                      0.01*m);

   logicalvolume_cherenkovSensitiveDetector = new G4LogicalVolume(solid_cherenkovSensitiveDetector,
                                                         world_material,
                                                         "logicalvolume_cherenkovSensitiveDetector");

   for (G4int iCnt = 0 ; iCnt < 100 ; iCnt++)
   {
       for (G4int jCnt = 0 ; jCnt < 100; jCnt++)
       {
          physicsvolume_cherenkovSenstiveDetector = new G4PVPlacement(0, 
                                                              G4ThreeVector(-0.5*m + (iCnt+0.5)*m/100, 
                                                              -0.5*m + (jCnt+0.5)*m/100, 
                                                              0.49*m),
                                                              logicalvolume_cherenkovSensitiveDetector, 
                                                              "physicsvolume_cherenkovSenstiveDetector", 
                                                              logicalvolume_world,
                                                              false,
                                                              jCnt + iCnt * 100, 
                                                              true);
       }
   }

}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  G4NistManager* nist_manager = G4NistManager::Instance();
  world_material = nist_manager->FindOrBuildMaterial("G4_AIR");

  G4double energy[2] = {1.239841939*eV/0.9, 1.239841939*eV/0.2};
  G4double rindexAerogel[2] = {1.1, 1.1};

  G4MaterialPropertiesTable *mpt_world = new G4MaterialPropertiesTable();
  mpt_world->AddProperty("RINDEX", 
                           energy, 
                           rindexAerogel, 
                           2);

  world_material->SetMaterialPropertiesTable(mpt_world);

  // Define detector world geometry
  solid_world = new G4Box("SolidWorld",             // its name
                           xworld, 
                           yworld, 
                           zworld);  // its size

  logicalvolume_world = new G4LogicalVolume(solid_world, 
                                            world_material, 
                                            "LogicalVolumeWorld");

  physicalvolume_world = new G4PVPlacement(nullptr,
                                           G4ThreeVector(0.0, 
                                                         0.0, 
                                                         0.0),
                                           logicalvolume_world,
                                           "PhysicalVolumeWorld",
                                           0,
                                           false,
                                           0,
                                           true);

  if (is_cherenkov)
  {
    this->cherenkov_squareGeometry();
    this->cherenkov_sensitiveDetector();
  }
  if (is_positronEmissionTomography)
  {
    this->PET_geometry();
    this->PET_sensitiveDetector();
  }


  return physicalvolume_world;
}

void DetectorConstruction::PET_sensitiveDetector()
{

  solid_PETdetector = new G4Box("solid_PETdetector", 
                              1*cm, 
                              5*cm, 
                              6*cm);

  logicalvolume_PETdetector = new G4LogicalVolume(solid_PETdetector, 
                                               world_material, 
                                               "logicalvolume_PETdetector");

  for (G4int i = 0; i < 6; i++)
  {
      for (G4int j = 0; j < 16; j++)
      {
          G4Rotate3D rotation_aboutZ(j* 22.5*deg, G4ThreeVector(0,0,1) );
          G4Translate3D translate_xdetector(G4ThreeVector(5.0/tan(22.5/2*deg)*cm + 6.0*cm + 5.0*cm, 
                                                         0*cm,
                                                         -40*cm + i * 15 * cm));

          G4Transform3D transform_detector = (rotation_aboutZ) * (translate_xdetector);

          physicalvolume_PETdetector = new G4PVPlacement(transform_detector,
                                                      logicalvolume_PETdetector,
                                                      "physicalvolume_PETdetector",
                                                      logicalvolume_world,
                                                      false,
                                                      0,
                                                      true);


      }
  }


}

void DetectorConstruction::PET_geometry()
{
  solid_scintillator = new G4Box("solid_scintillator",
                                  5*cm, 
                                  5*cm,
                                  6*cm);

  logicalvolume_scintillator = new G4LogicalVolume(solid_scintillator,
                                                   NaI,
                                                   "logicalvolume_scintillator");

  logicalvolume_scoring = logicalvolume_scintillator;

  for (G4int i = 0; i < 6; i++)
  {
      for (G4int j = 0; j < 16; j++)
      {
          G4Rotate3D rotation_aboutZ(j* 22.5*deg, G4ThreeVector(0,0,1) );
          G4Translate3D translate_xscintilator(G4ThreeVector(5.0/tan(22.5/2*deg)*cm + 5.0*cm, 
                                                             0*cm,
                                                             -40*cm + i * 15 * cm));

          G4Transform3D transform_scintilator = (rotation_aboutZ) * (translate_xscintilator);

          physicalvolume_scintillator = new G4PVPlacement(transform_scintilator,
                                                          logicalvolume_scintillator,
                                                          "physicalvolume_scintillator",
                                                          logicalvolume_world,
                                                          false,
                                                          0,
                                                          true);



      }
  }

}

void DetectorConstruction::material_NaI()
{
  G4NistManager* nist_manager = G4NistManager::Instance();
  NaI = new G4Material("NaI", 3.67*g/cm3, 2);
  Na = nist_manager->FindOrBuildElement("Na");
  I = nist_manager->FindOrBuildElement("I");
  NaI->AddElement(Na, 1);
  NaI->AddElement(I, 1);
}


// Construct the Sensitive Ditector Field.
void DetectorConstruction::ConstructSDandField()
{
  SensitiveDetector * sensitive_detector = new SensitiveDetector("nuce427lab02_sensitive_detector");

  if (is_cherenkov)
    logicalvolume_cherenkovSensitiveDetector->SetSensitiveDetector(sensitive_detector);
  else if (is_positronEmissionTomography)
    logicalvolume_PETdetector->SetSensitiveDetector(sensitive_detector);

  G4SDManager::GetSDMpointer()->SetVerboseLevel(1);
}

}

