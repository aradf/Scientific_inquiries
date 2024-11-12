#ifndef PRIMARY_GENERATOR_HH
#define PRIMARY_GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"

class G4Event;
class G4ParticleGun;

namespace NUCE427LAB03
{
class CLab03DetectorConstructor;

class CLab03PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
   CLab03PrimaryGeneratorAction( CLab03DetectorConstructor* detector_constructor);
   virtual ~CLab03PrimaryGeneratorAction();
   virtual void GeneratePrimaries(G4Event * event);

   void update_position();
   
private:
   void set_default();
 
   CLab03DetectorConstructor* fdetector_constructor;
   G4ParticleGun* fgun_particle;
};

}

#endif
