#ifndef RADIOACTIVE_DECAY_DETECTOR_CONSTRUCTION_HH
#define RADIOACTIVE_DECAY_DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4NistManager.hh"
#include "globals.hh"


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
   G4double get_worldsize() 
   {
      return fworld_size; 
   }
   
private:
   G4double fworld_size;
   /**
    * World Parameters.
    */
   G4Box *  solid_world;
   G4LogicalVolume *  logic_world;
   G4VPhysicalVolume * physical_world;

};


#endif