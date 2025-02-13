#ifndef DETECTOR_CONSTRUCTOR_HH
#define DETECTOR_CONSTRUCTOR_HH

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4VPhysicalVolume;
class G4Material;
class G4LogicalVolume;
class G4Box;
class G4Element;
class G4Tubs;
class G4VisAttributes;

namespace NUCE427LAB04
{
class CLab04DetectorConstructor : public G4VUserDetectorConstruction
{
public: 
    CLab04DetectorConstructor();
    ~CLab04DetectorConstructor();

    G4LogicalVolume* getscoring_logicalVolume() const
    {
        return fscoring_logicalVolume;
    }

    virtual G4VPhysicalVolume* Construct();
private:
    virtual void ConstructSDandField();

    G4Box* world_solid;

    G4LogicalVolume* world_logialVolume;
    G4LogicalVolume* fscoring_logicalVolume;

    G4VPhysicalVolume* world_physicalVolume;

    G4Material* world_material;

    void define_materials();
    void construct_scintillator();

    G4double xworld;
    G4double yworld;
    G4double zworld;

    G4Tubs* scintillator_solid;
    G4LogicalVolume* scintillator_logicalVolume;
    G4VPhysicalVolume* scintillator_physicalVolume;
    G4Material* NaI_material;
    G4Element* Na_element;
    G4Element* I_element;

    G4VisAttributes* green;

};


}

#endif
