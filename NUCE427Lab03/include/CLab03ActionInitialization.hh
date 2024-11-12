#ifndef ACTION_INITIALIZAION_HH
#define ACTION_INITIALIZAION_HH

#include "G4VUserActionInitialization.hh"

namespace NUCE427LAB03
{

class CLab03DetectorConstructor;

class CLab03ActionInitialization : public G4VUserActionInitialization
{
public:   
   CLab03ActionInitialization(CLab03DetectorConstructor* detector_constructor);
   virtual ~CLab03ActionInitialization();

   virtual void Build() const;
   virtual void BuildForMaster() const;

private:
   CLab03DetectorConstructor* fdetector_constructor;

};

}

#endif
