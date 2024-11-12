#include "CLab03EventAction.hh"
#include "CLab03Run.hh"

#include "G4RunManager.hh"
#include "G4Event.hh"


namespace NUCE427LAB03
{

CLab03EventAction::CLab03EventAction() : G4UserEventAction()
{
   this->fenergy_depositePerEvent = 0.0;
}
   
CLab03EventAction::~CLab03EventAction()
{

}

void CLab03EventAction::BeginOfEventAction(const G4Event* an_event)
{
   this->fenergy_depositePerEvent = 0.0;
}
   
void CLab03EventAction::EndOfEventAction(const G4Event* an_event)
{
   if (fenergy_depositePerEvent > 0.0 )
   {
      G4cout << " --> End of Event Action:  "
             << an_event->GetEventID()
             << ","
             << this->fenergy_depositePerEvent
             << G4endl;
   }  

  /*
   * Static cast performs data conversion.  Pointers of child class are converted to parent class.
   * The static cast operator does not check for success.
   */
  CLab03Run* current_run = static_cast< CLab03Run* > (G4RunManager::GetRunManager()->GetNonConstCurrentRun());
  if (current_run != nullptr)
  {
     current_run->fill_energyDepositInTarget( fenergy_depositePerEvent );
  }
  
   
}

}
