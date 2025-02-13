#ifndef RUN_ACTION_HH
#define RUN_ACTION_HH

#include <G4UserRunAction.hh>

namespace NUCE427LAB04
{

class CLab04RunAction : public G4UserRunAction
{
public:
    CLab04RunAction();
    ~CLab04RunAction();

    virtual void BeginOfRunAction(const G4Run* current_run);
    virtual void EndOfRunAction(const G4Run* current_run);

    virtual G4Run* GenerateRun();

private:


};


}



#endif
