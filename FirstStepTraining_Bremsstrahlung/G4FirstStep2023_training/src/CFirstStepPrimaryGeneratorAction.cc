#include "G4ParticleGun.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"

#include "CFirstStepPrimaryGeneratorAction.hh"
#include "CFirstStepDetectorConstruction.hh"

namespace FS
{

CFirstStepPrimaryGeneratorAction::CFirstStepPrimaryGeneratorAction(CFirstStepDetectorConstruction * detector_construction) 
: G4VUserPrimaryGeneratorAction()
{
   this->fdetector_construction = detector_construction;
   this->fgun_particle = nullptr; 
   G4int nparticle = 1;
   fgun_particle = new G4ParticleGun( nparticle );
   this->set_default();
}

CFirstStepPrimaryGeneratorAction::~CFirstStepPrimaryGeneratorAction()
{
   if (fgun_particle != nullptr)   
   {
      delete fgun_particle;
      fgun_particle = nullptr;
   }
}


void CFirstStepPrimaryGeneratorAction::set_default()
{
   /* Set the e- to be the particle definition */
   G4ParticleDefinition * particle_electron = G4ParticleTable::GetParticleTable()->FindParticle( "e-" );
   fgun_particle->SetParticleDefinition( particle_electron );
   fgun_particle->SetParticleMomentumDirection( G4ThreeVector( 1.0, 
                                                               0.0, 
                                                               0.0) );
   fgun_particle->SetParticleEnergy( 30.0 * CLHEP::MeV );

   /*
    * Position: ask the detector constructor for the x position.
    */
   this->update_position();
}

void CFirstStepPrimaryGeneratorAction::update_position()
{
   /*
    * Position: Ask the detetor to give an x-position update.
    */
   fgun_particle->SetParticlePosition ( G4ThreeVector ( fdetector_construction->get_gunXPosition(), 
                                                        0.0, 
                                                        0.0   ));

}

void CFirstStepPrimaryGeneratorAction::GeneratePrimaries(G4Event * event)
{
   G4int curent_eventid = event->GetEventID();
   // this->update_position();
   fgun_particle->GeneratePrimaryVertex( event );
}


}
