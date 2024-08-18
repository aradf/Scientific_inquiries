#ifndef PRIMARY_GENERATOR_HH
#define PRIMARY_GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"

class G4ParticleGun;
class G4Event;

namespace FS
{
class CFirstStepDetectorConstruction;

class CFirstStepPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
   CFirstStepPrimaryGeneratorAction(CFirstStepDetectorConstruction * detector_constructor);
   virtual ~CFirstStepPrimaryGeneratorAction();
   virtual void GeneratePrimaries(G4Event * event);

   void set_default();
   void update_position();
   
private:
   CFirstStepDetectorConstruction * fdetector_construction;
   G4ParticleGun * fgun_particle;

};

}
#endif