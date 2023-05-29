#ifndef RADIOACTIVE_DECAY_RUN1_HH
#define RADIOACTIVE_DECAY_RUN1_HH

#include "G4Run.hh"
#include "G4VProcess.hh"
#include "globals.hh"
#include <map>
#include "G4AnalysisManager.hh"

class G4ParticleDefinition;
class CRadioactiveDecayRun : public G4Run
{
  public:
    CRadioactiveDecayRun();
   ~CRadioactiveDecayRun();

  public:
    void particle_count(G4String, G4double, G4double);
    void balance(G4double, G4double);
    void event_timing(G4double);
    void primary_timing(G4double);
    void evis_event(G4double);

    void set_timewindow(G4double , G4double);
    void count_intimewindow(G4String, G4bool,G4bool,G4bool);
        
    void set_primary(G4ParticleDefinition* particle, G4double energy);
    void EndOfRun(); 

    virtual void Merge(const G4Run*);

  private:    
    struct particle_data 
    {
     particle_data() : fcount(0), fe_mean(0.), fe_min(0.), fe_max(0.), ft_mean(-1.) {}
     particle_data(G4int count, 
                  G4double ekin, 
                  G4double emin, 
                  G4double emax, 
                  G4double meanLife) : fcount(count), fe_mean(ekin), fe_min(emin), fe_max(emax), ft_mean(meanLife) {}
     G4int     fcount;
     G4double  fe_mean;
     G4double  fe_min;
     G4double  fe_max;
     G4double  ft_mean;
    };
     
  private: 
    G4ParticleDefinition*  fParticle;
    G4double  fEkin;
             
    std::map<G4String, particle_data>  fParticleDataMap;    
    G4int    fDecayCount, fTimeCount;
    G4double fEkinTot[3];
    G4double fPbalance[3];
    G4double fEventTime[3];
    G4double fPrimaryTime;
    G4double fEvisEvent[3];

private:    
  struct activity_data 
  {
      activity_data() : fn_life_t1(0), 
                        fn_life_t2(0), 
                        fn_decay_t1t2(0) {}

      activity_data(G4int n1, 
                    G4int n2, 
                    G4int nd) : fn_life_t1(n1), 
                                fn_life_t2(n2), 
                                fn_decay_t1t2(nd) {}
      G4int  fn_life_t1;
      G4int  fn_life_t2;
      G4int  fn_decay_t1t2;
  };
  
  std::map<G4String,activity_data>  fActivityMap;
  G4double fTimeWindow1, fTimeWindow2;
};
#endif

