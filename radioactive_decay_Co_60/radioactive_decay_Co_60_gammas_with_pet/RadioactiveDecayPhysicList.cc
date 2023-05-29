#include "RadioactiveDecayPhysicList.hh"

CRadioactiveDecayPhysicsList::CRadioactiveDecayPhysicsList()
{
   /**
    * The G4EmStandardPhysics manages ElectroMangetic Physics.
    * Not Hydronic Interaction.  It contains standard electromagnetic
    * process and radioactiveDecay module for generic ion.
    * The G4OpticalPhsics manages sentilation light (optical photons.)
    * 
    */

   RegisterPhysics (new G4EmStandardPhysics());
   RegisterPhysics (new G4OpticalPhysics());
   RegisterPhysics (new G4DecayPhysics());
   RegisterPhysics (new G4RadioactiveDecayPhysics());
}

CRadioactiveDecayPhysicsList::~CRadioactiveDecayPhysicsList()
{

   
}
