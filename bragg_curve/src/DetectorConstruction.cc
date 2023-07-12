#include "DetectorConstruction.hh"

#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4NistManager.hh>
#include <G4SystemOfUnits.hh>
#include <G4VisAttributes.hh>
#include <G4Box.hh>
#include <G4Orb.hh>
#include <G4SDManager.hh>
#include "G4MultiFunctionalDetector.hh"
#include "G4PSEnergyDeposit.hh"

#include <sstream>

using namespace std;

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4NistManager* nist = G4NistManager::Instance();
    G4double worldSizeX = 2 * m;
    G4double worldSizeY = 1 * m;
    G4double worldSizeZ = 1 * m;

    // We have created the world volume for you
    // As with all volumes, it requires three steps:

    // 1) Solid
    G4VSolid* worldBox = new G4Box("world", worldSizeX / 2, worldSizeY / 2, worldSizeZ / 2);

    // 2) Logical volume
    G4LogicalVolume* worldLog = new G4LogicalVolume(worldBox, nist->FindOrBuildMaterial("G4_AIR"), "world");
    G4VisAttributes* visAttr = new G4VisAttributes();
    visAttr->SetVisibility(false);
    worldLog->SetVisAttributes(visAttr);

    // 3) Physical volume
    G4VPhysicalVolume* worldPhys = new G4PVPlacement(nullptr, {}, worldLog, "world", nullptr, false, 0);

    
    // Create two instances of G4Element to support the "Water" material
    // (name, symbol, effectiveZ, effectiveA as mass/mole)
    G4Element* elH = new G4Element("Hydrogen", "H", 1., 1.0079 * g/mole);
    G4Element* elO = new G4Element("Carbon", "O", 8., 16.011 * g/mole);

    // Once you have the elements, create the material
    // (name, density, number of components)
    G4Material* water = new G4Material("Water", 1. * g/cm3, 2);
    water -> AddElement(elH, 2);
    water -> AddElement(elO, 1);

    // We have already provided the thickness of the water tank slices for you.
    G4double thickness = 0.5 * mm; 
    G4double width = 10 * cm;     
    G4double height = 10 * cm;    

    // Task 5: Create the solid volume for the water tank    
    G4VSolid* waterTankBox = new G4Box("waterTank", thickness / 2, width / 2, height / 2);
    
    
    // Task 5: Create a logical volume for the water tank
    G4LogicalVolume* waterTankLog = new G4LogicalVolume(waterTankBox, water,"waterTank"); 
    G4VisAttributes* blue = new G4VisAttributes(G4Colour::Blue());
    
    // Task 5: Colorize the water tank using proper visualisation  attributes
    blue -> SetVisibility(true);
    blue -> SetForceSolid(true);
    waterTankLog->SetVisAttributes(blue);

    // We have already provided the positions, you finish the rest... 
    G4int numberOfLayers = 100;
    G4double minX = 50 * cm + thickness / 2;

    vector<G4ThreeVector> waterTankPositions;

    for (int i = 0; i < numberOfLayers; i++)
    {
      waterTankPositions.push_back({minX + i * thickness, 0, 0});
    }

    for (int i = 0; i < numberOfLayers; i++)
    {
        ostringstream sName; sName << "waterTank" << i;
       
        // Task 5: Create 100 slices inside the water tank 
	      //  (use sName.str() for name)
        new G4PVPlacement(nullptr, waterTankPositions[i], waterTankLog, sName.str(), worldLog, 0, i);
    }

    //Show the material table
    G4cout << "Show the material table" << G4endl;
    G4cout << *(G4Material::GetMaterialTable()) << G4endl;

    // The Construct() method has to return the final (physical) world volume:
    return worldPhys;
}


void DetectorConstruction::ConstructSDandField()
{
    G4SDManager* sdManager = G4SDManager::GetSDMpointer();
    sdManager->SetVerboseLevel(2); 

    // Task 5.1: Create one instances of G4MultiFunctionalDetector (for the water tank)
    G4MultiFunctionalDetector* waterTankDetector = new G4MultiFunctionalDetector("waterTankLog");

    // Task 5.1: Create one primitive scorer for the dose and assign it to respective detectors
    G4VPrimitiveScorer* waterTankScorer = new G4PSEnergyDeposit("energy");
    waterTankDetector->RegisterPrimitive(waterTankScorer);

    // Task 5.1 Assign multi-functional detectors to the logical volumes and register them to the SDmanager
    SetSensitiveDetector("waterTank", waterTankDetector);
    sdManager->AddNewDetector(waterTankDetector);
   }
