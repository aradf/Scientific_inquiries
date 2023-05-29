#ifndef RADIOACTIVE_DECAY_DETECTOR_CONSTRUCTION_HH
#define RADIOACTIVE_DECAY_DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4GenericMessenger.hh"

#include "RadioactiveDecaySensitiveDetector.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalSkinSurface.hh"


/**
 * Class description 
 * This is inherits from the G4VUserDetectorConstruction which is a 
 * mandtory initialization class for detecter setup.  This class is the
 * place where the user describes the entire detector setup, including 
 * it's geometry, the material used in the construction, the definition 
 * of the sensitive region, and the readout schemes of the sensitive region.
 */
class CRadioactiveDecayDetectorConstruction : public G4VUserDetectorConstruction
{
public: 

   CRadioactiveDecayDetectorConstruction();
   ~CRadioactiveDecayDetectorConstruction();

   /**
    * The one pure virutal method is implemented and it is invoked by
    * G4RunManager when it's initialize method is invoked.  The 
    * construct method return the G4VPhysicalVolume pointer which 
    * represents the world volume.  In addition, the material and geometry 
    * are also described here.
    */
   virtual G4VPhysicalVolume * Construct();
   G4LogicalVolume * get_scoring_volume() const {return fscoring_volume; }
   
private:
   /**
    *  Implement this virtual method and assign magnetic field to volumes / regions.
    *  Define sensitive detectors and assign them to volumes
    */
   virtual void ConstructSDandField();
   void construct_scintillator();
   void define_material();

   /**
    * world definition.
    */
   G4Material * world_material;
   G4Box * world_solid;
   G4LogicalVolume * world_logic;
   G4VPhysicalVolume * world_physics;
   G4double xworld;
   G4double yworld;
   G4double zworld;

   /**
    * Scintillator definition.
    */
   G4Material * NaI;
   G4Element * C, * Na, * I;    
   G4Box * scintillator_solid;
   G4LogicalVolume * scintillator_logic;
   G4VPhysicalVolume * scintillator_physics;
   G4LogicalVolume * fscoring_volume;

   /**
    * detector definition.
    */
   G4Box * detector_solid;
   G4LogicalVolume  * detector_logic; 
   G4VPhysicalVolume * detector_physical;
   
   G4OpticalSurface * mirror_surface;
};


#endif