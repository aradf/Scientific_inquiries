#ifndef RADIOACTIVE_DECAY_PHYSICS_HH
#define RADIOACTIVE_DECAY_PHYSICS_HH

#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4OpticalPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"

/**
 * the particles to be used in the simulation,
 * the physics processes to be simulated.
 */
class CRadioactiveDecayPhysicsList : public G4VModularPhysicsList
{
public:
   CRadioactiveDecayPhysicsList();
   ~CRadioactiveDecayPhysicsList();
};


#endif