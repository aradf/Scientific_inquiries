#ifndef ANALYSIS_HH
#define ANALYSIS_HH

#include <G4RootAnalysisManager.hh>

// Use ROOT as output format for all Geant4 analysis tools
using G4AnalysisManager = G4RootAnalysisManager;

/**
 * comment the conent of the next two lines.

#include <G4CsvAnalysisManager.hh>
using G4AnalysisManager = G4CsvAnalysisManager;
 */

#endif
