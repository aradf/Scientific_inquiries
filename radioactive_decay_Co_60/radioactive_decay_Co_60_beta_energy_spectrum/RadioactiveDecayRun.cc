#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4PhysicalConstants.hh"

#include "RadioactiveDecayRun.hh"

CRadioactiveDecayRun::CRadioactiveDecayRun() : G4Run(), fParticle(0), fEkin(0.),  fDecayCount(0), fTimeCount(0), fPrimaryTime(0.), fTimeWindow1(0.), fTimeWindow2(0.)
{
  fEkinTot[0] = fPbalance[0] = fEventTime[0] = fEvisEvent[0] = 0. ;
  fEkinTot[1] = fPbalance[1] = fEventTime[1] = fEvisEvent[1] = DBL_MAX;
  fEkinTot[2] = fPbalance[2] = fEventTime[2] = fEvisEvent[2] = 0. ;
}

CRadioactiveDecayRun::~CRadioactiveDecayRun()
{ 


}

void CRadioactiveDecayRun::set_primary(G4ParticleDefinition* particle, G4double energy)
{ 
  fParticle = particle;
  fEkin = energy;
} 

void CRadioactiveDecayRun::particle_count(G4String name, G4double Ekin, G4double meanLife)
{
  std::map<G4String, particle_data>::iterator it = fParticleDataMap.find(name);
  if ( it == fParticleDataMap.end()) 
  {
    fParticleDataMap[name] = particle_data(1, Ekin, Ekin, Ekin, meanLife);
  }
    else 
    {
      particle_data& data = it->second;
      data.fcount++;
      data.fe_mean += Ekin;
      //update min max
      G4double emin = data.fe_min;
      if (Ekin < emin) data.fe_min = Ekin;
      G4double emax = data.fe_max;
      if (Ekin > emax) data.fe_max = Ekin;
      data.ft_mean = meanLife;
    }   

}

void CRadioactiveDecayRun::set_timewindow(G4double t1, G4double t2)
{
  fTimeWindow1 = t1;
  fTimeWindow2 = t2;
}

void CRadioactiveDecayRun::count_intimewindow(G4String name, 
                                                G4bool life1,
                                                G4bool life2, 
                                                G4bool decay)
{
  std::map<G4String, activity_data>::iterator it = fActivityMap.find(name);
  if ( it == fActivityMap.end()) 
  {
    G4int n1(0), n2(0), nd(0);
    if(life1) n1 = 1;
    if(life2) n2 = 1;
    if(decay) nd = 1;
    fActivityMap[name] = activity_data(n1, n2, nd);
  }
  else {
          activity_data& data = it->second;
          if(life1) data.fn_life_t1++;
          if(life2) data.fn_life_t2++;
          if(decay) data.fn_decay_t1t2++;
       }
}

void CRadioactiveDecayRun::balance(G4double Ekin, G4double Pbal)
{
  fDecayCount++;
  fEkinTot[0] += Ekin;
  //update min max  
  if (fDecayCount == 1) 
      fEkinTot[1] = fEkinTot[2] = Ekin;
  if (Ekin < fEkinTot[1]) 
      fEkinTot[1] = Ekin;
  if (Ekin > fEkinTot[2]) 
      fEkinTot[2] = Ekin;
  
  fPbalance[0] += Pbal;
  //update min max   
  if (fDecayCount == 1) 
      fPbalance[1] = fPbalance[2] = Pbal;  
  if (Pbal < fPbalance[1]) 
      fPbalance[1] = Pbal;
  if (Pbal > fPbalance[2]) 
      fPbalance[2] = Pbal;    
}

void CRadioactiveDecayRun::event_timing(G4double time)
{
  fTimeCount++;  
  fEventTime[0] += time;
  if (fTimeCount == 1) 
     fEventTime[1] = fEventTime[2] = time;  
  if (time < fEventTime[1]) 
     fEventTime[1] = time;
  if (time > fEventTime[2]) 
     fEventTime[2] = time;             
}

void CRadioactiveDecayRun::primary_timing(G4double ptime)
{
    fPrimaryTime += ptime;
}

void CRadioactiveDecayRun::evis_event(G4double Evis)
{
    fEvisEvent[0] += Evis;
    if (fTimeCount == 1) 
        fEvisEvent[1] = fEvisEvent[2] = Evis;  
    if (Evis < fEvisEvent[1]) 
        fEvisEvent[1] = Evis;
    if (Evis > fEvisEvent[2]) 
        fEvisEvent[2] = Evis;             
}

void CRadioactiveDecayRun::Merge(const G4Run* run)
{
      const CRadioactiveDecayRun* localRun = static_cast<const CRadioactiveDecayRun*>(run);

    //primary particle info
    //
    fParticle = localRun->fParticle;
    fEkin     = localRun->fEkin;
    
    // accumulate sums
    //
    fDecayCount  += localRun->fDecayCount;
    fTimeCount   += localRun->fTimeCount;  
    fPrimaryTime += localRun->fPrimaryTime;

    fEkinTot[0]   += localRun->fEkinTot[0];
    fPbalance[0]  += localRun->fPbalance[0];
    fEventTime[0] += localRun->fEventTime[0];
    fEvisEvent[0] += localRun->fEvisEvent[0];  
    
    G4double min,max;  
    min = localRun->fEkinTot[1]; max = localRun->fEkinTot[2];
    if (fEkinTot[1] > min) 
      fEkinTot[1] = min;
    if (fEkinTot[2] < max) 
      fEkinTot[2] = max;
    //
    min = localRun->fPbalance[1]; 
    max = localRun->fPbalance[2];
    
    if (fPbalance[1] > min) 
      fPbalance[1] = min;
    if (fPbalance[2] < max) 
      fPbalance[2] = max;
    
    //
    min = localRun->fEventTime[1]; 
    max = localRun->fEventTime[2];
    if (fEventTime[1] > min) 
      fEventTime[1] = min;
    if (fEventTime[2] < max) 
      fEventTime[2] = max;
    //
    min = localRun->fEvisEvent[1]; 
    max = localRun->fEvisEvent[2];
    
    if (fEvisEvent[1] > min) 
      fEvisEvent[1] = min;
    if (fEvisEvent[2] < max) 
      fEvisEvent[2] = max;
    
    //maps
    std::map<G4String, particle_data>::const_iterator itn;
    for (itn = localRun->fParticleDataMap.begin(); 
        itn != localRun->fParticleDataMap.end(); 
        ++itn) 
    {
      G4String name = itn->first;
      const particle_data& localData = itn->second;   
      if ( fParticleDataMap.find(name) == fParticleDataMap.end()) 
      {
        fParticleDataMap[name] = particle_data(localData.fcount, 
                                              localData.fe_mean, 
                                              localData.fe_min, 
                                              localData.fe_max, 
                                              localData.ft_mean);
      }
      else 
        {
            particle_data& data = fParticleDataMap[name];   
            data.fcount += localData.fcount;
            data.fe_mean += localData.fe_mean;
            G4double emin = localData.fe_min;
            
            if (emin < data.fe_min) 
              data.fe_min = emin;
            
            G4double emax = localData.fe_max;
            
            if (emax > data.fe_max) 
              data.fe_max = emax;
            
            data.ft_mean = localData.ft_mean;
        }   
    }
    
    //activity
    fTimeWindow1 = localRun->fTimeWindow1;
    fTimeWindow2 = localRun->fTimeWindow2;
    
    std::map<G4String, activity_data>::const_iterator ita;
    for (ita = localRun->fActivityMap.begin(); 
        ita != localRun->fActivityMap.end(); 
        ++ita) 
    {
      
      G4String name = ita->first;
      const activity_data& localData = ita->second;   
      if ( fActivityMap.find(name) == fActivityMap.end()) 
      {
        fActivityMap[name] = activity_data(localData.fn_life_t1, 
                                          localData.fn_life_t2, 
                                          localData.fn_decay_t1t2);
      } else {
                activity_data& data = fActivityMap[name];   
                data.fn_life_t1 += localData.fn_life_t1;
                data.fn_life_t2 += localData.fn_life_t2;
                data.fn_decay_t1t2 += localData.fn_decay_t1t2;
              }
    }
    
    G4Run::Merge(run); 
} 
    
void CRadioactiveDecayRun::EndOfRun() 
{
    G4int nbEvents = numberOfEvent;
    G4String partName = fParticle->GetParticleName();
    
    G4cout << "\n ======================== run summary ======================";  
    G4cout << "\n The run was " << nbEvents << " " << partName << " of "
            << G4BestUnit(fEkin,"Energy");
    G4cout << "\n ===========================================================\n";
    G4cout << G4endl;
    
    if (nbEvents == 0) 
    { 
        return; 
    }
    
    G4int prec = 4, wid = prec + 2;
    G4int dfprec = G4cout.precision(prec);
          
    //particle count
    G4cout << " Nb of generated particles: \n" << G4endl;
        
    std::map<G4String, particle_data>::iterator it;               
    for (it = fParticleDataMap.begin(); 
          it != fParticleDataMap.end(); 
          it++) 
    { 
        G4String name     = it->first;
        particle_data data = it->second;
        G4int count    = data.fcount;
        G4double eMean = data.fe_mean/count;
        G4double eMin  = data.fe_min;
        G4double eMax  = data.fe_max;
        G4double meanLife = data.ft_mean;
            
        G4cout << "  " << std::setw(15) << name << ": " << std::setw(7) << count
              << "  Emean = " << std::setw(wid) << G4BestUnit(eMean, "Energy")
              << "\t( "  << G4BestUnit(eMin, "Energy")
              << " --> " << G4BestUnit(eMax, "Energy") << ")";

        if (meanLife > 0.)
          G4cout << "\tmean life = " << G4BestUnit(meanLife, "Time")   << G4endl;
        else if (meanLife < 0.) 
          G4cout << "\tstable" << G4endl;
        else 
          G4cout << G4endl;
    }
    
    //energy momentum balance
    if (fDecayCount > 0) 
    {
        G4double Ebmean = fEkinTot[0]/fDecayCount;
        G4double Pbmean = fPbalance[0]/fDecayCount;
            
        G4cout << "\n   Ekin Total (Q single decay): mean = "
              << std::setw(wid) << G4BestUnit(Ebmean, "Energy")
              << "\t( "  << G4BestUnit(fEkinTot[1], "Energy")
              << " --> " << G4BestUnit(fEkinTot[2], "Energy")
              << ")" << G4endl;    
              
        G4cout << "\n   Momentum balance (excluding gamma desexcitation): mean = " 
              << std::setw(wid) << G4BestUnit(Pbmean, "Energy")
              << "\t( "  << G4BestUnit(fPbalance[1], "Energy")
              << " --> " << G4BestUnit(fPbalance[2], "Energy")
              << ")" << G4endl;
    }
                
    //total time of life
    if (fTimeCount > 0) 
    {
        G4double Tmean = fEventTime[0]/fTimeCount;
        G4double halfLife = Tmean*std::log(2.);
      
        G4cout << "\n   Total time of life (full chain): mean = "
              << std::setw(wid) << G4BestUnit(Tmean, "Time")
              << "  half-life = "
              << std::setw(wid) << G4BestUnit(halfLife, "Time")
              << "   ( "  << G4BestUnit(fEventTime[1], "Time")
              << " --> "  << G4BestUnit(fEventTime[2], "Time")
              << ")" << G4endl;
    }

    //total visible Energy
    if (fTimeCount > 0) {
        G4double Evmean = fEvisEvent[0]/fTimeCount;
      
        G4cout << "\n   Total visible energy (full chain) : mean = "
              << std::setw(wid) << G4BestUnit(Evmean,  "Energy")
              << "   ( "  << G4BestUnit(fEvisEvent[1], "Energy")
              << " --> "  << G4BestUnit(fEvisEvent[2], "Energy")
              << ")" << G4endl;
    }

    //activity of primary ion
    G4double pTimeMean = fPrimaryTime/nbEvents;
    G4double molMass = fParticle->GetAtomicMass()*g/mole;
    G4double nAtoms_perUnitOfMass = Avogadro/molMass;
    G4double Activity_perUnitOfMass = 0.0;
    if (pTimeMean > 0.0)
    { 
        Activity_perUnitOfMass = nAtoms_perUnitOfMass/pTimeMean; 
    }
      
    G4cout << "\n   Activity of " << partName << " = "
                << std::setw(wid)  << Activity_perUnitOfMass*g/becquerel
                << " Bq/g   ("     << Activity_perUnitOfMass*g/curie
                << " Ci/g) \n" 
                << G4endl;
        
    //activities in time window
    if (fTimeWindow2 > 0.) 
    {
      G4double dt = fTimeWindow2 - fTimeWindow1;
      G4cout << "   Activities in time window [t1, t2] = [" 
              << G4BestUnit(fTimeWindow1, "Time") << ", "
              << G4BestUnit(fTimeWindow2, "Time") << "]  (delta time = "
              << G4BestUnit(dt, "Time") << ") : \n" << G4endl;

      std::map<G4String, activity_data>::iterator ita;               
      for (ita = fActivityMap.begin(); 
            ita != fActivityMap.end(); 
            ita++) 
      { 
          G4String name     = ita->first;
          activity_data data = ita->second;
          G4int n1     = data.fn_life_t1;
          G4int n2     = data.fn_life_t2;
          G4int ndecay = data.fn_decay_t1t2;
          G4double actv = ndecay/dt;

          G4cout << "  " << std::setw(15) << name << ": "
                << "  n(t1) = " << std::setw(7) << n1
                << "\tn(t2) = " << std::setw(7) << n2
                << "\t   decays = " << std::setw(7) << ndecay 
                << "   ---> <actv> = "  << G4BestUnit(actv, "Activity") << "\n";
      }
    }
    G4cout << G4endl;
    
    //normalize histograms
    G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();
    G4double factor = 100./nbEvents;
    analysisManager->ScaleH1(1,factor);
    analysisManager->ScaleH1(2,factor);
    analysisManager->ScaleH1(3,factor);
    analysisManager->ScaleH1(4,factor);
    analysisManager->ScaleH1(5,factor);
                                                    
    // remove all contents in fParticleDataMap
    fParticleDataMap.clear();
    fActivityMap.clear();

    // restore default precision
    G4cout.precision(dfprec);
}
