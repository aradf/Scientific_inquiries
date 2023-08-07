#ifndef PHYSICS_LIST_HH
#define PHYSICS_LIST_HH

#include <G4VModularPhysicsList.hh>

namespace B1
{

class PhysicsList : public G4VModularPhysicsList
{
public:
  PhysicsList();
  ~PhysicsList(){;};

  //! Optional virtual methods, to gain direct control on 
  //! the particle/processes definition. Not used here

  //! Mandatory method 
  void 	SetCuts ();

};

}

#endif
