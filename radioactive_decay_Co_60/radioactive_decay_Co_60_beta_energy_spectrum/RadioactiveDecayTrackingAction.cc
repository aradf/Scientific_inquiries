#include "RadioactiveDecayTrackingAction.hh"
#include "RadioactiveDecayRun.hh"
#include "G4RunManager.hh"
#include "G4ParticleTypes.hh"

CRadioactiveDecayTrackingAction::CRadioactiveDecayTrackingAction(CRadioactiveDecayEventAction* event_action) : G4UserTrackingAction(), fevent(event_action)
{
   G4String hello_world;

}

CRadioactiveDecayTrackingAction::~CRadioactiveDecayTrackingAction()
{


}
   
void CRadioactiveDecayTrackingAction::PreUserTrackingAction(const G4Track* some_track)
{
    CRadioactiveDecayRun* run = static_cast<CRadioactiveDecayRun*>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());
    G4ParticleDefinition* particle = some_track->GetDefinition();
    G4String name   = particle->GetParticleName();
    fcharge = particle->GetPDGCharge();
    fmass   = particle->GetPDGMass();
    G4double kinetic_energy = some_track->GetKineticEnergy();
    G4int ID      = some_track->GetTrackID();
    // check LifeTime
    G4double mean_life = particle->GetPDGLifeTime();
    
    //count particles
    run->particle_count(name, kinetic_energy, mean_life);

    //energy spectrum
    G4int ih = 0;
    if (particle == G4Electron::Electron()|| 
        particle == G4Positron::Positron())
    {
        ih = 1;
    }
    else if (particle == G4NeutrinoE::NeutrinoE()|| 
            particle == G4AntiNeutrinoE::AntiNeutrinoE()) 
            {
                ih = 2;
            }

    if (ih) 
    {
        G4AnalysisManager::Instance()->FillH1(ih, kinetic_energy);
    }

/*
  ----- Run* run = static_cast<Run*>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());
  ----- G4ParticleDefinition* particle = track->GetDefinition();
  ----- G4String name   = particle->GetParticleName();
  ----- fCharge = particle->GetPDGCharge();
  ----- fMass   = particle->GetPDGMass();  
  ----- G4double Ekin = track->GetKineticEnergy();
  ----- G4int ID      = track->GetTrackID();
  G4bool condition = false;
  
  ----- // check LifeTime
  ----- G4double meanLife = particle->GetPDGLifeTime();
  
  ----- //count particles
  ----- run->ParticleCount(name, Ekin, meanLife);
  
  -----   //energy spectrum
  -----   G4int ih = 0;
  ----- if (particle == G4Electron::Electron()|| 
  -----       particle == G4Positron::Positron())
  -----   {
  -----      ih = 1;
  -----   }
  -----   else if (particle == G4NeutrinoE::NeutrinoE()|| 
  -----            particle == G4AntiNeutrinoE::AntiNeutrinoE()) 
  -----           {
  -----              ih = 2;
  -----           }
  else if (particle == G4Gamma::Gamma()) 
          {
            ih = 3;
          }
  else if (particle == G4Alpha::Alpha()) 
          {
            ih = 4;
          }
  else if (fCharge > 2.) 
          {
            ih = 5;
          }
  
  -----  if (ih) 
  -----  {
  -----    G4AnalysisManager::Instance()->FillH1(ih, Ekin);
  -----  }
  
  //Ion
  //
  if (fCharge > 2.) 
  {
    //build decay chain
    if (ID == 1) 
      fEvent->AddDecayChain(name);
    else       
      fEvent->AddDecayChain(" ---> " + name);
    
    //full chain: put at rest; if not: kill secondary      
    G4Track* tr = (G4Track*) track;
    if (fFullChain) 
    { 
        tr->SetKineticEnergy(0.);
        tr->SetTrackStatus(fStopButAlive);
    }
    else if (ID>1)
          {
             tr->SetTrackStatus(fStopAndKill);
          }
    //
    fTime_birth = track->GetGlobalTime();
  }
  
  //example of saving random number seed of this fEvent, under condition
  //
  ////condition = (ih == 3);
  if (condition) 
    G4RunManager::GetRunManager()->rndmSaveThisEvent();
*/


}

void CRadioactiveDecayTrackingAction::PostUserTrackingAction(const G4Track* some_track)
{

  //keep only ions
  if (fcharge < 3.0 ) 
    return;

/*
  ----- //keep only ions
  -----   if (fCharge < 3. ) 
  -----     return;
  
  Run* run = static_cast<Run*>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());
   
  G4AnalysisManager* analysis = G4AnalysisManager::Instance();
  
  //get time
  G4double time = track->GetGlobalTime();
  G4int ID = track->GetTrackID();
  
  if (ID == 1) 
      run->PrimaryTiming(time);        //time of life of primary ion
  
  fTime_end = time;
      
  //energy and momentum balance (from secondaries)
  //
  const std::vector<const G4Track*>* secondaries = track->GetStep()->GetSecondaryInCurrentStep();
  size_t nbtrk = (*secondaries).size();
  if (nbtrk) 
  {
    //there are secondaries --> it is a decay
    //
    //balance    
    G4double EkinTot = 0., EkinVis = 0.;
    G4ThreeVector Pbalance = - track->GetMomentum();
  
    for (size_t itr=0; itr<nbtrk; itr++) 
    {
       const G4Track* trk = (*secondaries)[itr];
       G4ParticleDefinition* particle = trk->GetDefinition();
       G4double Ekin = trk->GetKineticEnergy();
       EkinTot += Ekin;
       G4bool visible = !((particle == G4NeutrinoE::NeutrinoE())||
                          (particle == G4AntiNeutrinoE::AntiNeutrinoE()));

       if (visible) 
        EkinVis += Ekin; 
       //exclude gamma desexcitation from momentum balance
       if (particle != G4Gamma::Gamma()) 
        Pbalance += trk->GetMomentum();
    }
    
    G4double Pbal = Pbalance.mag();  
    run->Balance(EkinTot,Pbal);  
    analysis->FillH1(6,EkinTot);
    analysis->FillH1(7,Pbal);
    fEvent->AddEvisible(EkinVis);
  }
  
  //no secondaries --> end of chain    
  //  
  if (!nbtrk) 
  {
    run->EventTiming(time);                     //total time of life
    G4double weight = track->GetWeight();
    analysis->FillH1(8,time,weight);
    ////    analysis->FillH1(8,time);    
    fTime_end = DBL_MAX;
  }
  
  //count activity in time window
  run->SetTimeWindow(fTimeWindow1, fTimeWindow2);
  
  G4String name   = track->GetDefinition()->GetParticleName();
  G4bool life1(false), life2(false), decay(false);
  if ((fTime_birth <= fTimeWindow1) && (fTime_end > fTimeWindow1)) 
    life1 = true;

  if ((fTime_birth <= fTimeWindow2)&&(fTime_end > fTimeWindow2)) 
    life2 = true;

  if ((fTime_end   >  fTimeWindow1)&&(fTime_end < fTimeWindow2)) 
    decay = true;

  if (life1||life2||decay) 
    run->CountInTimeWindow(name,life1,life2,decay);
*/

}
