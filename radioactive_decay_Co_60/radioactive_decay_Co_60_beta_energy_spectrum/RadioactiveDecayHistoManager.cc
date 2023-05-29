#include "RadioactiveDecayHistoManager.hh"
#include "G4UnitsTable.hh"

CRadioactiveDecayHistoManager::CRadioactiveDecayHistoManager() : ffile_name("decay_sim")
{
  configure();
}

CRadioactiveDecayHistoManager::~CRadioactiveDecayHistoManager()
{
}

void CRadioactiveDecayHistoManager::configure()
{
  /** 
   * Create or get analysis manager
  */
  G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();
  analysis_manager->SetDefaultFileType("root");
  analysis_manager->SetFileName(ffile_name);
  analysis_manager->SetVerboseLevel(1);

  /** 
   * enable inactivation of histograms
  */
  analysis_manager->SetActivation(true);     

  /** 
   * Define histograms start values
  */
  const G4int kmax_histo = 10;
  const G4String id[] = {"0","1","2","3","4","5","6","7","8","9"};
  const G4String title[] = 
          { "dummy",                                    //0
            "energy spectrum (%): e+ e-",               //1
            "energy spectrum (%): nu_e anti_nu_e",      //2
            "energy spectrum (%): gamma",               //3
            "energy spectrum (%): alpha",               //4
            "energy spectrum (%): ions",                //5
            "total kinetic energy per single decay (Q)",//6
            "momentum balance",                         //7
            "total time of life of decay chain",        //8
            "total visible energy in decay chain"       //9
          };

  /** 
   * Default values (to be reset via /analysis/h1/set command)
   */
  G4int nbins = 100;
  G4double vmin = 0.;
  G4double vmax = 100.;

  /** 
   * Create all histograms as inactivated as we have not yet set nbins, vmin, vmax
   */
  for (G4int k=0; k<kmax_histo; k++) 
  {
    G4int ih = analysis_manager->CreateH1( id[k], 
                                           title[k], 
                                           nbins, 
                                           vmin, 
                                           vmax);
    analysis_manager->SetH1Activation( ih, 
                                       false);
  }

}


