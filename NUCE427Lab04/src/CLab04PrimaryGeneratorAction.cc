#include <CLab04PrimaryGeneratorAction.hh>

#include <G4ParticleGun.hh>
#include <G4Event.hh>
#include <G4ParticleTable.hh>
#include <G4ParticleDefinition.hh>
#include <G4Geantino.hh>
#include <G4SystemOfUnits.hh>
#include <G4IonTable.hh>

namespace NUCE427LAB04
{

CLab04PrimaryGeneratorAction::CLab04PrimaryGeneratorAction()
{
    G4cout << "INFO: CLab04PrimaryGeneratorAction Constructor ..."
           << G4endl;
    fparticle_gun = nullptr;
    fparticle_gun = new G4ParticleGun( 1 );

    G4ParticleTable* particle_table = G4ParticleTable::GetParticleTable();

    G4String particle_name = "geantino";
    G4ParticleDefinition* particle_definition = particle_table->FindParticle( particle_name );

    G4ThreeVector particle_position( 0.0, 0.0, 0.0 );
    G4ThreeVector particle_momentum( 0.0, 0.0, 1.0 );
    fparticle_gun->SetParticlePosition( particle_position );
    fparticle_gun->SetParticleMomentumDirection( particle_momentum );
    fparticle_gun->SetParticleMomentum( 0.0 * CLHEP::GeV);
    fparticle_gun->SetParticleDefinition ( particle_definition );


}
    
CLab04PrimaryGeneratorAction::~CLab04PrimaryGeneratorAction()
{
    if (fparticle_gun != nullptr)
    {
        delete fparticle_gun;
        fparticle_gun = nullptr;
    }
}    

void CLab04PrimaryGeneratorAction::GeneratePrimaries(G4Event* current_event)
{
#warning "This proton particle definition must be replaced with Cs-137 or Au-198 "
    G4ParticleDefinition* particle_definition = fparticle_gun->GetParticleDefinition();
    if (particle_definition == G4Geantino::Geantino())
    {
        // 198-Au defined.
        G4int Z = 79;
        G4int A = 198;

        // 137-Cs defined.
        // G4int Z = 55;
        // G4int A = 137;

        G4double charge = 0.0 * CLHEP::eplus;
        G4double energy = 0.0 * CLHEP::keV;

        G4ParticleDefinition* ion = G4IonTable::GetIonTable()->GetIon( Z, A, energy );
        fparticle_gun->SetParticleDefinition( ion );
        fparticle_gun->SetParticleCharge( charge );
    }

    fparticle_gun->GeneratePrimaryVertex( current_event );
}

    
}