//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************

#include "CHistogramManager.hh"
#include "G4UnitsTable.hh"


CHistogramManager::CHistogramManager() : ffile_name("TungstenCharacteristicRadiation")
{
   Book();
}

CHistogramManager::~CHistogramManager()
{

}

void CHistogramManager::Book()
{
   /*
    * Create or get analysis manager
    * The choice of analysis technology is done via selection of a namespace in HistogramManger.hh
    */
   G4AnalysisManager* analysis_manager = G4AnalysisManager::Instance();
#warning "remove csv to root"  
   /*
    * analysisManager->SetDefaultFileType("root");
    */
   analysis_manager->SetDefaultFileType("csv");
   analysis_manager->SetFileName(ffile_name);
   analysis_manager->SetVerboseLevel(1);
   analysis_manager->SetActivation(true);   //enable inactivation of histograms
  
   /*
    * Define histograms start values
    */
   const G4int kmax_histogram = 7;
   const G4String id[] = { "0", "1", "2", "3" , "4", "5", "6"};
   const G4String title[] = { "dummy",                                              //0
                              "scattered primary particle: energy spectrum",        //1
                              "scattered primary particle: costheta distribution",  //2
                              "charged secondaries: energy spectrum",               //3
                              "charged secondaries: costheta distribution",         //4
                              "neutral secondaries: energy spectrum",               //5
                              "neutral secondaries: costheta distribution"          //6
                            };  

   /*
    * Default values (to be reset via /analysis/h1/set command)
    * Content of the mac file.
    * /analysis/h1/set 3 50000 0.001 100.0 keV	#energy  of e-  
    * /analysis/h1/set 5 50000 0.001 100.0 keV	#energy  of gamma  
    */
   G4int number_bins = 100;
   G4double value_min = 0.0;
   G4double value_max = 100.0;

   /*
    * Create all histograms as inactivated as we have not yet set nbins, vmin, vmax
    */
   for (G4int kCnt = 0; kCnt < kmax_histogram; kCnt++) 
   {
      G4int ih = analysis_manager->CreateH1( id[kCnt], 
                                             title[kCnt], 
                                             number_bins, 
                                             value_min, 
                                             value_max);

      analysis_manager->SetH1Activation(ih, false);
   }
}

