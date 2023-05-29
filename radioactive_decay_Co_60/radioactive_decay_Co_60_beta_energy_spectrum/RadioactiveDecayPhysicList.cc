#include "RadioactiveDecayPhysicList.hh"
#include "G4UnitsTable.hh"
#include "G4ParticleTypes.hh"
#include "G4IonConstructor.hh"
#include "G4PhysicsListHelper.hh"
#include "G4Radioactivation.hh"
#include "G4SystemOfUnits.hh"
#include "G4NuclideTable.hh"
#include "G4LossTableManager.hh"
#include "G4UAtomicDeexcitation.hh"
#include "G4NuclideTable.hh"
#include "G4NuclearLevelData.hh"
#include "G4DeexPrecoParameters.hh"
#include "G4PhysListUtil.hh"
#include "G4EmBuilder.hh"
#include "globals.hh"

CRadioactiveDecayPhysicsList::CRadioactiveDecayPhysicsList(): G4VUserPhysicsList()
{
  /**
   * instantiate Physics List infrastructure
   */
  G4PhysListUtil::InitialiseParameters();


  /**
   * Update G4NuclideTable time limit
   * G4NuclideTable is provided as a list of nuclei in GEANT4. It
   * contains about 2900 ground states and 4000 excited states
   */
  const G4double mean_life = 1*picosecond;  
  G4NuclideTable::GetInstance()->SetMeanLifeThreshold(mean_life);  
  G4NuclideTable::GetInstance()->SetLevelTolerance(1.0*eV);

  /**
   * Define flags for the atomic de-excitation module.
   */
  G4EmParameters::Instance()->SetDefaults();
  G4EmParameters::Instance()->SetAugerCascade(true);
  G4EmParameters::Instance()->SetDeexcitationIgnoreCut(true);    

  /**
   * Define flags for nuclear gamma de-excitation model
   */
  G4DeexPrecoParameters* deex = G4NuclearLevelData::GetInstance()->GetParameters();
  deex->SetCorrelatedGamma(false);
  deex->SetStoreAllLevels(true);
  deex->SetInternalConversionFlag(true);	  
  deex->SetIsomerProduction(true);  
  deex->SetMaxLifeTime(mean_life);

  /**
   * Set default cut in range value
   */
  SetDefaultCutValue(1*mm);
}

CRadioactiveDecayPhysicsList::~CRadioactiveDecayPhysicsList()
{

   
}

/**
 *  This is a pure virtual method, in which the static member functions for all the particles you
 *  require should be called. This ensures that objects of these particles are created. 
 */
void CRadioactiveDecayPhysicsList::ConstructParticle() 
{
  /**
   * Minimal set of particles for EM physics and radioactive decay
   */
  G4EmBuilder::ConstructMinimalEmSet();
}

/**
 *  physics processes should be created and registered with each particle’s
 *  instance of G4ProcessManager.
 * 
 */
void CRadioactiveDecayPhysicsList::ConstructProcess() 
{
  /**
   *  Define transportation process.  The G4Transportation class (and/or related classes) 
   *  describes the particle motion in space and time. It is the mandatory process for 
   *  tracking particles.
   */
  AddTransportation();
  
  G4Radioactivation* radioactiveDecay = new G4Radioactivation();

  /**
   * Atomic Rearangement
   */
  G4bool ARM_flag = false;
  radioactiveDecay->SetARM(ARM_flag);        
  	  
  /**
   * EM physics constructor is not used in this example, so 
   * it is needed to instantiate and initialize atomic de-excitation.
   */
  G4LossTableManager* manager = G4LossTableManager::Instance();
  G4VAtomDeexcitation* deex = manager->AtomDeexcitation();
  if (nullptr == deex) 
  {
     deex = new G4UAtomicDeexcitation();
     manager->SetAtomDeexcitation(deex);
  }
  deex->InitialiseAtomicDeexcitation();

  /**
   * Register RadioactiveDecay. The registration of the relevant Transportation 
   * process is handled by the G4PhysicsListHelper, which chooses the correct 
   * type depending on whether any of the features which require parallel geometries 
   * have been used.
   */
  G4PhysicsListHelper* physics_listhelper = G4PhysicsListHelper::GetPhysicsListHelper();
  physics_listhelper->RegisterProcess(radioactiveDecay, 
                                      G4GenericIon::GenericIon());
}
