#ifndef CHERENKOV_SENSITIVE_DETECTOR_HH
#define CHERENKOV_SENSITIVE_DETECTOR_HH

#include "G4VSensitiveDetector.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4AnalysisManager.hh"

/**
 * Class description
 * This class represents the detector(s).  The hit objects must be 
 * constructed from the steps along the paritlc track.  
 */
class CCherenkovSensitiveDetector : public G4VSensitiveDetector
{
public: 
    CCherenkovSensitiveDetector(G4String );
    ~CCherenkovSensitiveDetector();

private:
   /**
    * Construct the hit objects from the steps along the paritlc track.
    * This method is invoked by G4SteppngManager when a step is composed in
    * the G4Step object.  
    * Note: G4TouchableHistory is obsolete and not used.
    * The Geant4 developer MUST implement this method for generating hit(s)
    * utilizing the members and methods of G4Step object(s).  
    * Note: The volume and position members and methods are available in 
    * PreStepPoint of G4Step.  In addition, the G4TouchableHistory object 
    * of the tracking geometry is stored in the PreStepPoint object.
    * Please keep in mind that this method is proteced and SHALL be invoked 
    * inside the Hit() method in the base class
    * after readout geometry associated to the senstive detector is handled.  
    */
   virtual G4bool ProcessHits(G4Step *, G4TouchableHistory * );

};

#endif