#ifndef PHYSICS_LIST_HH
#define PHYSICS_LIST_HH

#include  "G4VModularPhysicsList.hh"

namespace NUCE427LAB03
{
class CLab03PhysicsList : public G4VModularPhysicsList
{
public:
   CLab03PhysicsList();
   ~CLab03PhysicsList();

   virtual void ConstructParticle();
   virtual void ConstructProcess();
   virtual void SetCuts();
private:

};

}

#endif