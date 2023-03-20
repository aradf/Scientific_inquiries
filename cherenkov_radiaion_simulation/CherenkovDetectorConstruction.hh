#ifndef CHERENKOV_DETECTOR_CONSTRUCTION_HH
#define CHERENKOV_DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4GenericMessenger.hh"

#include "CherenkovSensitiveDetector.hh"

/**
 * Class description 
 * This is inherits from the G4VUserDetectorConstruction which is a 
 * mandtory initialization class for detecter setup.  
 */

class CCherenkovDetectorConstruction : public G4VUserDetectorConstruction
{
public: 

   CCherenkovDetectorConstruction();
   ~CCherenkovDetectorConstruction();
   void DefineMaterial();

   G4LogicalVolume * get_scoring_volume() const {return fscoring_volume; }

   /**
    * The one pure virutal method is implemented and it is invoked by
    * G4RunManager when it's initialize method is invoked.  The 
    * construct method return the G4VPhysicalVolume pointer which 
    * represents the world volume.
    */
   virtual G4VPhysicalVolume * Construct();

private:
   /**
    *  Implement this virtual method and assign magnetic field to volumes / regions.
    *  Define sensitive detectors and assign them to volumes
    */
   virtual void ConstructSDandField();
 
   /**
    * The world's geometry
    */
   G4int number_columns;
   G4int number_rows;
   G4double xworld;
   G4double yworld;
   G4double zworld;

   /**
    * The solid's geomerty
    */
   G4Box * solid_world;
   G4Box * solid_radiator;
   G4Box * solid_detector;

   /**
    * The logic volume geomerty
    */
   G4LogicalVolume * logic_world;
   G4LogicalVolume * logic_radiator;
   G4LogicalVolume * logic_detector;

   /**
    * The physical volume geomerty
    */
   G4VPhysicalVolume * physics_detector;
   G4VPhysicalVolume * physics_radiator;
   G4VPhysicalVolume * physics_world;

   /**
    * The material definition.
    */
   G4Material * SiO2;
   G4Material * H2O;
   G4Material * Aerogel;
   G4Material * world_material;
   G4Element  * C;

   G4GenericMessenger * fmessenger;
   G4LogicalVolume * fscoring_volume;
   
};


#endif
