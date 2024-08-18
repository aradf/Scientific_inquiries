#ifndef ACTION_INITIALIZAION_HH
#define ACTION_INITIALIZAION_HH

#include "G4VUserActionInitialization.hh"

namespace FS
{
    
class CFirstStepDetectorConstruction;

class CFirstStepActionInitialization : public G4VUserActionInitialization
{
public:
   CFirstStepActionInitialization(CFirstStepDetectorConstruction *detector_construction);
   virtual ~CFirstStepActionInitialization();

   //Build method must have access to the dectector construction .
   virtual void Build() const;
   virtual void BuildForMaster() const;

private:
   CFirstStepDetectorConstruction * fdetector_construction;

};

}

#endif