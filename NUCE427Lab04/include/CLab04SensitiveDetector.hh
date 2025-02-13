#ifndef SENSITIVE_DETECTOR_HH
#define SENSITIVE_DETECTOR_HH

#include "G4VSensitiveDetector.hh"

namespace NUCE427LAB04
{

class CLab04SensitiveDetector : public G4VSensitiveDetector
{
public:
    CLab04SensitiveDetector(G4String );
    ~CLab04SensitiveDetector();

private:
    virtual G4bool ProcessHits(G4Step*, G4TouchableHistory *);

};


}

#endif
