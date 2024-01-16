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

#ifndef NUCE427LAB02DetectorConstruction_h
#define NUCE427LAB02DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "G4NistManager.hh"
#include "globals.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;
class G4Material;
class G4Box;
class G4Tubs;
class G4GenericMessenger;

namespace NUCE427LAB02
{
/// Detector construction class to define materials and geometry.
/// Crystals are positioned in Ring, with an appropriate rotation matrix.
/// Several copies of Ring are placed in the full detector.

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    ~DetectorConstruction();

  public:
    virtual G4VPhysicalVolume* Construct();
    virtual void ConstructSDandField();
    G4LogicalVolume * get_scoring_volume() const { return logicalvolume_scoring; }

  private:
    void cherenkov_squareGeometry();
    void cherenkov_sensitiveDetector();
    void PET_sensitiveDetector();
    void PET_geometry();
    void material_NaI();
    void material_aerogel();
    
    G4bool fCheckOverlaps = true;
    G4Material * aerogel;
    G4Material * SiO2;
    G4Material * H2O;
    G4Element  * C;
    G4Material * NaI;
    G4Element  * Na;
    G4Element  * I;


    G4Box * solid_cherenkovsquare;
    G4LogicalVolume * logicalvolume_cherenkovsquare;
    G4VPhysicalVolume * physicalvolume_cherenkovsquare;
    G4LogicalVolume * logicalvolume_scoring;

    G4Box * solid_cherenkovSensitiveDetector;
    G4LogicalVolume * logicalvolume_cherenkovSensitiveDetector;
    G4VPhysicalVolume * physicsvolume_cherenkovSenstiveDetector;

    G4Material * world_material;
    G4Box * solid_world;
    G4LogicalVolume * logicalvolume_world;
    G4VPhysicalVolume * physicalvolume_world;

    G4Box * solid_scintillator;
    G4LogicalVolume * logicalvolume_scintillator;
    G4VPhysicalVolume * physicalvolume_scintillator;

    G4Box * solid_PETdetector;
    G4LogicalVolume * logicalvolume_PETdetector;
    G4VPhysicalVolume * physicalvolume_PETdetector;

    G4GenericMessenger * fgeneric_messenger;
    G4int number_columns;
    G4int number_rows;
    G4double xworld;
    G4double yworld;
    G4double zworld;

    G4bool is_cherenkov;
    G4bool is_positronEmissionTomography;
};

}

#endif

