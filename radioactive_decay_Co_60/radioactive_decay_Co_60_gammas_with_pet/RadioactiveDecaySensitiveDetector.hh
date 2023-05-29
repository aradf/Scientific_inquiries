#ifndef RADIOACTIVE_DECAY_SENSITIVEDETECTOR_HH
#define RADIOACTIVE_DECAY_SENSITIVEDETECTOR_HH

#include "G4VSensitiveDetector.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4AnalysisManager.hh"

class CRadioactiveDecaySensitiveDetector : public G4VSensitiveDetector
{
public:
   CRadioactiveDecaySensitiveDetector(G4String );
   ~CRadioactiveDecaySensitiveDetector();

private:
   virtual G4bool ProcessHits(G4Step *, G4TouchableHistory *);

   G4PhysicsFreeVector * quantum_efficency;
   // G4PhysicsOrderedFreeVector * quantum_efficency;
};

#endif