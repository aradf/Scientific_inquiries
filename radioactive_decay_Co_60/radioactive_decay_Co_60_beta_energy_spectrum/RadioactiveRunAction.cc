#include "RadioactiveRunAction.hh"



/*
CRadioactiveDecayRunAction::CRadioactiveDecayRunAction(CRadioactiveDecayPrimaryGenerator * primary_generator) : G4UserRunAction(), 
                                                       fprimary(primary_generator), 
                                                       frun(0), 
                                                       fhistogram_manager(0)

*/

CRadioactiveDecayRunAction::CRadioactiveDecayRunAction(CRadioactiveDecayPrimaryGenerator * primary_generator) : G4UserRunAction(), 
                                                       fprimary(primary_generator),
                                                       fradioactivedecay_run(0), 
                                                       fradioactivedecay_histogrammanager(0)
{
  fradioactivedecay_histogrammanager = new CRadioactiveDecayHistoManager();
}

CRadioactiveDecayRunAction::~CRadioactiveDecayRunAction()
{
  delete fradioactivedecay_histogrammanager;
}

void CRadioactiveDecayRunAction::BeginOfRunAction(const G4Run * some_run)
{

  /**
   * keep run condition
   */

  if (fprimary) 
  { 
    G4ParticleDefinition* particle = fprimary->get_particlegun()->GetParticleDefinition();
    G4double energy = fprimary->get_particlegun()->GetParticleEnergy();
    fradioactivedecay_run->set_primary(particle, energy);
  }    

  /**
   * histograms
   */
  G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();
  if ( analysis_manager->IsActive() ) 
  {
     analysis_manager->OpenFile();
  }
}

void CRadioactiveDecayRunAction::EndOfRunAction(const G4Run *)
{

  /**
   * histograms
   */
 G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();
 if ( analysis_manager->IsActive() ) 
 {
   analysis_manager->Write();
   analysis_manager->CloseFile();
 } 

}

G4Run * CRadioactiveDecayRunAction::GenerateRun()
{ 
  fradioactivedecay_run = new CRadioactiveDecayRun();
  return fradioactivedecay_run;
}
