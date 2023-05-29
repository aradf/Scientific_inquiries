#ifndef RADIOACTIVE_DECAY_PHYSICS_HH
#define RADIOACTIVE_DECAY_PHYSICS_HH

#include "G4VUserPhysicsList.hh"

/**
 * the particles to be used in the simulation,
 * the physics processes to be simulated.
 * G4VUserPhysicsLit is the pure vitual base class for a physics list.
 */
class CRadioactiveDecayPhysicsList : public G4VUserPhysicsList
{
public:
   CRadioactiveDecayPhysicsList();
   ~CRadioactiveDecayPhysicsList();

protected:
   /**
   * Construct particle and physics. This is a pure virtual method and 
   * must be implemented. 
   */
   void ConstructParticle() override;
   void ConstructProcess() override;
};


#endif