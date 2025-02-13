#ifndef EVENT_ACTION_HH
#define EVENT_ACTION_HH

#include <G4UserEventAction.hh>

#include <CLab04RunAction.hh>

class G4Event;

namespace NUCE427LAB04
{

class CLab04EventAction : public G4UserEventAction
{
public:
    CLab04EventAction(CLab04RunAction* );
    ~CLab04EventAction();

    virtual void BeginOfEventAction(const G4Event* );
    virtual void EndOfEventAction(const G4Event* );

    void add_energyDeposit(G4double energy_deposit) 
    { 
        fenergy_deposit += energy_deposit;
    }

private:
    G4double fenergy_deposit;

};


}


#endif