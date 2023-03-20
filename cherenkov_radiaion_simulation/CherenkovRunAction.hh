#ifndef CHERENKOV_RUNACTION_HH
#define CHERENKOV_RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
// #include "g4root.hh"
#include "G4AnalysisManager.hh"

class CCherenkovRunAction : public G4UserRunAction
{
public:
   CCherenkovRunAction();
   ~CCherenkovRunAction();

   /**
    * This method is invoked at the beginning of BeamOn() method.
    * Set a run identification number, booking histograms, set run specific
    * conditions of the sensetive detector
    */
   virtual void BeginOfRunAction(const G4Run *);

   /**
    * This method is invoked at the very end of BeamOn() method.
    * store/print histrogram and manipulate run summaries.
    */
   virtual void EndOfRunAction(const G4Run *);
};



#endif 
