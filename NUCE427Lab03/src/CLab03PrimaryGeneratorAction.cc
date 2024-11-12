#include "G4ParticleGun.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"
#include "Randomize.hh"

#include "CLab03DetectorConstruction.hh"
#include "CLab03PrimaryGeneratorAction.hh"

namespace NUCE427LAB03
{

CLab03PrimaryGeneratorAction::CLab03PrimaryGeneratorAction( CLab03DetectorConstructor* detector_constructor) 
                                                            : G4VUserPrimaryGeneratorAction()
{
   /*
    * There is no need to perform memory allocation for the fdetector_constructor;
    */
   fdetector_constructor = detector_constructor;
   G4int number_particle = 1;
   fgun_particle = nullptr;
   fgun_particle = new G4ParticleGun( number_particle );
   set_default();
  
}

CLab03PrimaryGeneratorAction::~CLab03PrimaryGeneratorAction()
{
   if (fgun_particle != nullptr)   
   {
      delete fgun_particle;
      fgun_particle = nullptr;
   }

}

void CLab03PrimaryGeneratorAction::set_default()
{
   G4ParticleDefinition* particle_neutron = G4ParticleTable::GetParticleTable()->FindParticle("neutron");
   fgun_particle->SetParticleDefinition( particle_neutron );
   fgun_particle->SetParticleMomentumDirection( G4ThreeVector( 1.0, 
                                                               0.0, 
                                                               0.0) );
   // fgun_particle->SetParticleEnergy( 2.0 * CLHEP::MeV );
   update_position();
}

void CLab03PrimaryGeneratorAction::update_position()
{
   /*
    * TBD: Ask the detector to give x-position.
    */
   fgun_particle->SetParticlePosition( G4ThreeVector ( fdetector_constructor->get_gunXPosition(), 
                                                       0.0,
                                                       0.0 ) );

}

void CLab03PrimaryGeneratorAction::GeneratePrimaries(G4Event * some_event)
{
   G4int current_eventid = some_event->GetEventID();
   // update_position();
   
   /*
    * Add code for the partile energy probability density function (PDF)
    * INITIAL BEAM ENERGY
    */
   double particle_energy = 0.0;
   particle_energy = G4RandGauss::shoot( 2.0 * CLHEP::MeV,
                                         5.0955e-5 * CLHEP::MeV );

#warning "TBD: Remove the particle energy of 2.0 MeV for production.   "
   //particle_energy = 2.0 * CLHEP::MeV;

   fgun_particle->SetParticleEnergy( particle_energy );
   fgun_particle->GeneratePrimaryVertex( some_event );
}



} // end of namespace