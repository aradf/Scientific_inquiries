#ifndef DETECTOR_CONSTRUCTOR_HH
#define DETECTOR_CONSTRUCTOR_HH

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4VPhysicalVolume;
class G4Material;

namespace NUCE427LAB03
{

class CLab03DetectorConstructor : public G4VUserDetectorConstruction
{
public:
   CLab03DetectorConstructor();
   virtual ~CLab03DetectorConstructor();
   virtual G4VPhysicalVolume* Construct() override;
   const G4VPhysicalVolume* gettarget_physicalVolume() const
   {
      return ftarget_physicalVolume;
   }
   
   G4double get_gunXPosition() const
   {
      return fgun_xposition;
   }
   void set_targetThickness(const G4double target_thickness);
   
   G4double get_targetThickness()
   {
      return ftarget_thickness;
   }

private:
   G4Material* get_moderatorMaterial() const
   {
      return fmoderator_material;
   }
   void set_moderatorMaterial();

   const G4Material* get_shieldMaterial() const
   {
      return fshield_material;
   }
   void set_shieldMaterial();

   G4Material* get_targetMaterial() const
   {
      return ftarget_material;
   }
   void set_targetMaterial();

   G4double ftarget_thickness;
   /* Pointer variable to several  material */
   G4Material* fmoderator_material;
   G4Material* fshield_material;
   G4Material* ftarget_material;
   G4VPhysicalVolume* ftarget_physicalVolume;
   G4double fgun_xposition;

};
}

#endif