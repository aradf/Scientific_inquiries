#include "G4Event.hh"

#include "CFirstStepEventAction.hh"
#include "CFirstStepRun.hh"
#include "G4RunManager.hh"

namespace FS
{

CFirstStepEventAction::CFirstStepEventAction() : G4UserEventAction()
{
   this->fenergy_depositePerEvent = 0.0;
}

CFirstStepEventAction::~CFirstStepEventAction()
{


}

void CFirstStepEventAction::BeginOfEventAction(const G4Event * an_event)
{
    this->fenergy_depositePerEvent = 0.0;
}

void CFirstStepEventAction::EndOfEventAction(const G4Event * an_event)
{
    // G4cout << "   --> During Event ID: "
    //        << an_event->GetEventID()
    //        << " Added Energy Deposite during event "
    //        << this->fenergy_depositePerEvent
    //        << " (MeV). "
    //        << G4endl;
    // this->fenergy_depositePerEvent = 0.0;

    G4cout << an_event->GetEventID()
           << ","
           << this->fenergy_depositePerEvent
           << G4endl;

  /*
   * Static cast performs data conversion.  Pointers of child class are converted to parent class.
   * The static cast operator does not check for success.
   */
  CFirstStepRun * current_run = static_cast< CFirstStepRun * > ( G4RunManager::GetRunManager()->GetNonConstCurrentRun() );
  
  if ( current_run != nullptr)
  {
       current_run->fill_energyDepositInTarget(fenergy_depositePerEvent);
  }

}

}



