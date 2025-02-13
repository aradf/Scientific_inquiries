#include "CLab04PhysicsList.hh"

#include <G4EmStandardPhysics.hh>
// #include <G4OpticalPhysics.hh>

#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"



namespace NUCE427LAB04
{

CLab04PhysicsList::CLab04PhysicsList()
{
    G4cout << "INFO: CLab04PhysicsList Constructor ..."
           << G4endl;

    RegisterPhysics ( new G4EmStandardPhysics());
    // RegisterPhysics ( new G4OpticalPhysics());
    RegisterPhysics ( new G4DecayPhysics());
    RegisterPhysics ( new G4RadioactiveDecayPhysics());

}

CLab04PhysicsList::~CLab04PhysicsList()
{
    
}

}