#ifndef CHERENKOV_PHYSICS_HH
#define CHERENKOV_PHYSICS_HH

#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4OpticalPhysics.hh"

/**
 * the particles to be used in the simulation,
 * the physics processes to be simulated.
 */
class CCherenkovPhysicsList : public G4VModularPhysicsList
{
public:
   CCherenkovPhysicsList();
   ~CCherenkovPhysicsList();
};


#endif