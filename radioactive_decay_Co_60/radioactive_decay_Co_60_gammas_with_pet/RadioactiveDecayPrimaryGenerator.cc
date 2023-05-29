#include "RadioactiveDecayPrimaryGenerator.hh"


/**
 * The primary genertor(s) and the initial conditions are defined.
 */
CRadioactiveDecayPrimaryGenerator::CRadioactiveDecayPrimaryGenerator()
{

  /**
   * One primary particle per event is created.  One run contains serveral events.  
   * Each event can have multple particles.  This object generates primary particle(s)
   * with a particular momentum and position.  it does not provide any randomizing.  
   * The integer input value causes the generation of one or more primaries with exactly
   * same kinematics.  Geant4 provides varies methods To generate a primary with randomized 
   * energy, momentum, and/or position, random number generation and various distributions.
   */
   fparticle_gun = new G4ParticleGun(1);
   G4ParticleTable * particle_table = G4ParticleTable::GetParticleTable();
   G4ParticleDefinition * particle = particle_table->FindParticle("geantino");

   G4ThreeVector particle_pos(0.0, 
                              0.0, 
                              0.0);

   G4ThreeVector particle_mom(0.0, 
                              0.0, 
                              1.0);
                              
   fparticle_gun->SetParticlePosition(particle_pos);
   fparticle_gun->SetParticleMomentumDirection(particle_mom);
   fparticle_gun->SetParticleMomentum( 0.0 * GeV); 
   fparticle_gun->SetParticleDefinition(particle);       

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
   G4ParticleDefinition * particle = fparticle_gun->GetParticleDefinition();
   if (particle == G4Geantino::Geantino())
   {
       G4int Z = 27;
       G4int A = 60;
    
       G4double charge = 0.0*eplus;
       G4double energy = 0.0*keV;

       G4ParticleDefinition * ion = G4IonTable::GetIonTable()->GetIon(Z, A, energy);
       fparticle_gun->SetParticleDefinition(ion);
       fparticle_gun->SetParticleCharge(charge);
   }
   fparticle_gun->GeneratePrimaryVertex(an_event);
}
