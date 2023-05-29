#ifndef RADIOACTIVE_DECAY_ACTION_HH
#define RADIOACTIVE_DECAY_ACTION_HH

#include "G4VUserActionInitialization.hh"
#include "RadioactiveDecayPrimaryGenerator.hh"
#include "RadioactiveRunAction.hh"
#include "RadioactiveDecayEventAction.hh"
#include "RadioactiveDecaySteppingAction.hh"

/**
 * Class Description:
 * This class inherits from G4VuserActionInitialization which is the abstract
 * base class for instantiating all the user action classes.  It has to implemnt
 * the Build virutal method which is invoked by G4RunManager for sequential 
 * execution and G4WorkerRunManager.  Note that the parent virtual method 
 * 'build' is declared as constant.  It means objcts of CCherenkovActionInitialization 
 * can be constructed, however, should not store the pointers of these CCherenkovActionInitialization 
 * objects as data members of the CCherenkovActionInitialization class.
 */
class CRadioactiveDecayActionInitialization : public G4VUserActionInitialization
{
public:
  CRadioactiveDecayActionInitialization();
  ~CRadioactiveDecayActionInitialization();

/**
 * This virtual methos is implemnted in CCherenkovActionInitialization 
 * to instantiate user action class objects.
 */
virtual void Build() const;

};

#endif 
