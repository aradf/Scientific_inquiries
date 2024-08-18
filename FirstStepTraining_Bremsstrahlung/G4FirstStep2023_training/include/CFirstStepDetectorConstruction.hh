#ifndef G4FSDetectorConstruction_h
#define G4FSDetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;
class G4Material;

namespace FS
{

class CFirstStepDetectorConstruction : public G4VUserDetectorConstruction
{
public:
   CFirstStepDetectorConstruction();
   virtual ~CFirstStepDetectorConstruction();

   virtual G4VPhysicalVolume* Construct() override;

   void set_targetMaterial(const G4String& nist_material);

   const G4Material* get_targetMaterial() const {
      return ftarget_material; 
   }

   void set_targetThickness(const G4double target_thckness);

   const G4double get_targetThickness() const {
      return ftarget_thickness;
   }

   const G4VPhysicalVolume * get_targetPhysicalVolume() const {
      return ftarget_physicalVolume;
   }

   G4double get_gunXPosition() const {
      return fgun_xposition;
   }

private:
   /*
    * A pointer variable to the detector constructor provides the target's volume, target
    * material, target thickness and etc..
    */
   
   /* Pointer variable to the target material */
   G4Material * ftarget_material;

   /* l-value variable storing the the target thickness*/
   G4double ftarget_thickness;

   /* A pointer varialbe to the target physical volume.  ftarget_physicalVolume == 0x1234,
    * '*' operator  point to the content r-value of the class type G4VPhysicalVolume,
    * '->' arrow operator allows to invoked the class type methods and &ftarget_physicalVolume
    * == 0xABCD.  Note that the physical volume is simply a placed instance of the logical volume
    * and it must be placed inside world logical volume.
    */
   G4VPhysicalVolume * ftarget_physicalVolume;

   /* The proper x position fo the gun (primary particle generator)*/  
   G4double fgun_xposition;

};

}

#endif
