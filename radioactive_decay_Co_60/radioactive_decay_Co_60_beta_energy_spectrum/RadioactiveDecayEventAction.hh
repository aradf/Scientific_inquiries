#ifndef RADIOACTIVEDECAY_EVENTACTION_HH
#define RADIOACTIVEDECAY_EVENTACTION_HH 

#include "G4UserEventAction.hh"
#include "globals.hh"

/**
 * This is the base class of one of the user's optional action classes.
 * The two veritual methods BeginOfEventAction and EndOfEventAction are
 * invoked in the beginning and end of an event.
 */
class CRadioactiveDecayEventAction : public G4UserEventAction
{
public:
   CRadioactiveDecayEventAction();
   ~CRadioactiveDecayEventAction();

public:
    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);

    void add_decaychain(G4String value) {fdecay_chain += value;};
    void add_evisible(G4double value)   {fevisible_total    += value;};
    
  private:
    G4String        fdecay_chain;                   
    G4double        fevisible_total;
};

#endif