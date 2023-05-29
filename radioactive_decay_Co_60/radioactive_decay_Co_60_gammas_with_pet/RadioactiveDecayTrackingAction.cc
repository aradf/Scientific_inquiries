#include "RadioactiveDecayTrackingAction.hh"

#include "G4Track.hh"
#include "G4ParticleTypes.hh"

CRadioactiveDecayTrackingAction::CRadioactiveDecayTrackingAction(CRadioactiveDecayEventAction * event_action) : G4UserTrackingAction(), 
                                                                 fevent_action(event_action), 
                                                                 fcharge(0.0), 
                                                                 fmass(0.0)
{

}

CRadioactiveDecayTrackingAction::~CRadioactiveDecayTrackingAction()
{

}   
void CRadioactiveDecayTrackingAction::PreUserTrackingAction(const G4Track * current_track)
{
    G4cout << "PreUserTrackingAction : " << G4endl;
    G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();
    G4ParticleDefinition* particle = current_track->GetDefinition();
    G4String name  = particle->GetParticleName();
    fcharge = particle->GetPDGCharge();
    fmass   = particle->GetPDGMass();  

    G4double kinetic_energy = current_track->GetKineticEnergy();
    G4int track_id      = current_track->GetTrackID();
    G4bool condition = false;

    /* 
    * check LifeTime 
    */ 
    G4double meanLife = particle->GetPDGLifeTime();

    /* 
    * energy spectrum 
    */ 
    G4int ih = 0;
    if (particle == G4Electron::Electron()|| 
        particle == G4Positron::Positron())
    {
      ih = 1;
    }

    if (particle == G4Electron::Electron()|| 
        particle == G4Positron::Positron())
    {
      ih = 1;
    }


    if (ih && (kinetic_energy < 0.4)) 
    {
      // do something.
      // G4AnalysisManager::Instance()->FillH1(ih, Ekin);
      G4cout << "kinetic_energy : " << kinetic_energy << " [MeV] " << G4endl;
      G4double fESpec = kinetic_energy;
      manager_analysis ->FillNtupleDColumn(3, 0, fESpec);
      manager_analysis->AddNtupleRow(3);
    }

    /*
      G4ParticleDefinition* particle = track->GetDefinition();
      G4String name   = particle->GetParticleName();
      fCharge = particle->GetPDGCharge();
      fMass   = particle->GetPDGMass();  
        
      G4double Ekin = track->GetKineticEnergy();
      G4int ID      = track->GetTrackID();
      
      G4bool condition = false;
      
      // check LifeTime
      //
      G4double meanLife = particle->GetPDGLifeTime();
      
      //count particles
      //
      run->ParticleCount(name, Ekin, meanLife);
      
      //energy spectrum
      //
      G4int ih = 0;
      if (particle == G4Electron::Electron()|| 
          particle == G4Positron::Positron())
      {
        ih = 1;
      }

      if (ih) 
      {
        // do something.
        // G4AnalysisManager::Instance()->FillH1(ih, Ekin);
      }
    */  
}
    
    
void CRadioactiveDecayTrackingAction::PostUserTrackingAction(const G4Track * current_track)
{
  G4cout << "PostUserTrackingAction : " << G4endl;

}
    
