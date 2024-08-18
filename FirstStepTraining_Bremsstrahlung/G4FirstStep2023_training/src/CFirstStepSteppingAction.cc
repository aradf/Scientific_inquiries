#include "CFirstStepSteppingAction.hh"
#include "CFirstStepDetectorConstruction.hh"
#include "CFirstStepEventAction.hh"

#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"

namespace FS
{

CFirstStepSteppingAction::CFirstStepSteppingAction(CFirstStepDetectorConstruction * detector_construction,
                                                   CFirstStepEventAction * event_action)
                                                   : G4UserSteppingAction()
{
    this->fdetector_constructor = detector_construction;
    this->fevent_action = event_action;
}

CFirstStepSteppingAction::~CFirstStepSteppingAction()
{

}

void CFirstStepSteppingAction::UserSteppingAction(const G4Step * current_step)
{
   /* Pointer variable to properties before the current step */
   G4StepPoint * prestep_point = nullptr;
   G4VPhysicalVolume * current_physicalVolume = nullptr;
   const G4VPhysicalVolume * target_physicalVolume = nullptr;

   prestep_point = current_step->GetPreStepPoint();
   if (prestep_point != nullptr)
   {
        /* Get the physical volume of the current step*/
        current_physicalVolume = prestep_point->GetPhysicalVolume();
        target_physicalVolume = fdetector_constructor->get_targetPhysicalVolume();
   }

   // if (current_physicalVolume != nullptr)
   // {
   //    G4cout << "    --> Step done in: " 
   //           << current_physicalVolume->GetName()
   //           << " with Total Energy Deposite "   
   //           << current_step->GetTotalEnergyDeposit() / CLHEP::MeV
   //           << " (MeV) "
   //           << G4endl;
   // }

   /* Get the target physical volume and compare it against the current physical volume */
   if (current_physicalVolume == target_physicalVolume)
   {
      G4double total_energyDeposit = current_step->GetTotalEnergyDeposit() / CLHEP::MeV;
      
      // G4cout << "    --> Step done in: " 
      //        << current_physicalVolume->GetName()
      //        << " with Energy Deposite "   
      //        << total_energy
      //        << " (MeV) "
      //        << G4endl;

      if (total_energyDeposit > 0.0)
         (this->fevent_action)->add_energyDepsoitePerEvent( total_energyDeposit );
   }

   /* Get the current stacking track for the current step*/
   // G4Track * current_stack = current_step->GetTrack();
   


}

}    
