#include "PhysicsList.hh"

#include <G4EmStandardPhysics.hh>
#include <G4DecayPhysics.hh> 
#include <G4ProductionCutsTable.hh>
#include <G4SystemOfUnits.hh>
#include <G4EmLivermorePhysics.hh>
#include <G4EmExtraPhysics.hh>

#include <G4HadronPhysicsFTFP_BERT.hh>
#include <G4HadronElasticPhysics.hh>

PhysicsList::PhysicsList()
{
  // Standard EM physics 
  RegisterPhysics(new G4EmStandardPhysics());
  
  // Default Decay Physics
  RegisterPhysics(new G4DecayPhysics());
    
  RegisterPhysics(new G4EmExtraPhysics());

  RegisterPhysics(new G4HadronElasticPhysics());
  RegisterPhysics(new G4HadronPhysicsFTFP_BERT());
}


void PhysicsList::SetCuts()
{
  // The method SetCuts() is mandatory in the interface. Here, one just use 
  // the default SetCuts() provided by the base class.
  G4VUserPhysicsList::SetCuts();
  
  // G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(100*eV,100.*GeV);  
    
  // In addition, dump the full list of cuts for the materials used in 
  // the setup
  DumpCutValuesTable();
}
