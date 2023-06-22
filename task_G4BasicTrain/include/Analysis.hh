#ifndef ANALYSIS_HH
#define ANALYSIS_HH

#include <G4RootAnalysisManager.hh>
// Task 4c.4 Uncomment the line below.
#include <G4CsvAnalysisManager.hh>

// Use ROOT as output format for all Geant4 analysis tools
// Comment the line below.
//using G4AnalysisManager = G4RootAnalysisManager;

// Task 4c.4: Uncomment the line below
// one to swap between different output formats of g4analysis
// Use CSV as output format with all Geant4 analysis tools
//
using G4AnalysisManager = G4CsvAnalysisManager;

#endif
