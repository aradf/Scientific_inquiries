#include "RadioactiveDecayPrimaryGenerator.hh"
#include "G4SystemOfUnits.hh"
#include "G4IonTable.hh"
#include "G4Geantino.hh"

/**
 * The primary genertor(s) and the initial conditions are defined.
 */
CRadioactiveDecayPrimaryGenerator::CRadioactiveDecayPrimaryGenerator() : G4VUserPrimaryGeneratorAction(), fparticle_gun(0)
{

  /**
   * One primary particle per event is created.  One run contains serveral events.  
   * Each event can have multple particles.  This object generates primary particle(s)
   * with a particular momentum and position.  it does not provide any randomizing.  
   * The integer input value causes the generation of one or more primaries with exactly
   * same kinematics.  Geant4 provides varies methods To generate a primary with randomized 
   * energy, momentum, and/or position, random number generation and various distributions.
   */
   G4int n_particle = 1;
   fparticle_gun  = new G4ParticleGun(n_particle);

   fparticle_gun->SetParticleEnergy(0.0*eV);
   fparticle_gun->SetParticlePosition(G4ThreeVector(0.0,
                                                    0.0,
                                                    0.0));

   fparticle_gun->SetParticleMomentumDirection(G4ThreeVector(1.0,
                                                             0.0,
                                                             0.0));          
}

CRadioactiveDecayPrimaryGenerator::~CRadioactiveDecayPrimaryGenerator()
{
  delete fparticle_gun;
}

/**
 * Invoke the G4VPrimaryGenerator class already instantiated via the generatePrimaryVertex() method.
 */
void CRadioactiveDecayPrimaryGenerator::GeneratePrimaries(G4Event * an_event)
{
  if (fparticle_gun->GetParticleDefinition() == G4Geantino::Geantino()) 
  {  
    G4int Z = 10, A = 24;
    G4double ion_charge   = 0.*eplus;
    G4double excit_energy = 0.*keV;
    
    G4ParticleDefinition* ion = G4IonTable::GetIonTable()->GetIon(Z,A,excit_energy);
    fparticle_gun->SetParticleDefinition(ion);
    fparticle_gun->SetParticleCharge(ion_charge);
  }    
  /**
   * Create Vertex
   */
  fparticle_gun->GeneratePrimaryVertex(an_event);
}
