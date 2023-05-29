#include "RadioactiveDecayTrackingMessenger.hh"

CRadioActiveDecayTrackingMessenger::CRadioActiveDecayTrackingMessenger(CRadioactiveDecayTrackingAction * tracka):G4UImessenger(),
                                                                       ftracking_action(tracka),
                                                                       ftracking_cmd(0)
{


}

CRadioActiveDecayTrackingMessenger::~CRadioActiveDecayTrackingMessenger()
{



}

void CRadioActiveDecayTrackingMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
   if (command == ftracking_cmd)
   { 
        // ftracking_action->SetFullChain(fTrackingCmd->GetNewBoolValue(newValue));
   }

}
