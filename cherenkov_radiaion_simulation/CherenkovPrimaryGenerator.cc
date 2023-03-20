#include "CherenkovPrimaryGenerator.hh"


/**
 * The primary genertor(s) and the initial conditions are defined.
 */
CCherenkovPrimaryGenerator::CCherenkovPrimaryGenerator()
{

  /**
   * One primary particle per event is created.  One run contains serveral events.  
   * Each event can have multple particles.  This object generates primary particle(s)
   * with a particular momentum and position.  it does not provide any randomizing.  
   * The integer input value causes the generation of one or more primaries with exactly
   * same kinematics.  Geant4 provides varies methods To generate a primary with randomized 
   * energy, momentum, and/or position, random number generation and various distributions.
   */
  fparticle_gun  = new G4ParticleGun(1);

  /**
   * The type of particle are defined here.  The G4ParticleTable is a singlton and 
   * the pointer to this object is provided by GetParticleTable method.
   */
  G4ParticleTable * particle_table = G4ParticleTable::GetParticleTable();
  G4String particle_name = "proton";

  G4ParticleDefinition * particle_definition = particle_table->FindParticle(particle_name);

  G4ThreeVector particle_position(0.0,
                                  0.0,
                                  0.0);

  G4ThreeVector particle_momentum(0.0,
                                  0.0,
                                  1.0);

  fparticle_gun->SetParticlePosition(particle_position);
  fparticle_gun->SetParticleMomentumDirection(particle_momentum);
  fparticle_gun->SetParticleMomentum(100.0*GeV);
  fparticle_gun->SetParticleDefinition(particle_definition);
}

CCherenkovPrimaryGenerator::~CCherenkovPrimaryGenerator()
{
  delete fparticle_gun;
}

/**
 * Invoke the G4VPrimaryGenerator class already instantiated via the generatePrimaryVertex() method.
 */
void CCherenkovPrimaryGenerator::GeneratePrimaries(G4Event * an_event)
{
  fparticle_gun->GeneratePrimaryVertex(an_event);
}
