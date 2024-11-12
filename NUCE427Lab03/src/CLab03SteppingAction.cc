#include "CLab03SteppingAction.hh"
#include "CLab03DetectorConstruction.hh"
#include "CLab03EventAction.hh"

#include "G4StepPoint.hh"
#include "G4Step.hh"
#include "G4VPhysicalVolume.hh"

namespace NUCE427LAB03
{

CLab03SteppingAction::CLab03SteppingAction( CLab03DetectorConstructor* detector_construction,
                                            CLab03EventAction* event_action ) : G4UserSteppingAction()
{
   this->fdetector_constructor = detector_construction;
   this->fevent_action = event_action;
}

CLab03SteppingAction::~CLab03SteppingAction()
{

}
   
void CLab03SteppingAction::UserSteppingAction( const G4Step* current_step)
{
   /* Pointer variable to properties before the current step */
   G4StepPoint* prestep_point = nullptr;
   G4VPhysicalVolume* current_physicalVolume = nullptr;
   const G4VPhysicalVolume* target_physicalVolume = nullptr;

   prestep_point = current_step->GetPreStepPoint();
   if (prestep_point != nullptr)
   {
      /* Get the physical volume of the current step*/
      current_physicalVolume = prestep_point->GetPhysicalVolume();
      target_physicalVolume = fdetector_constructor->gettarget_physicalVolume();
      // G4cout << "    --> Step done in: " 
      //        << current_physicalVolume->GetName()
      //        << " with Energy Deposite "   
      //        << current_step->GetTotalEnergyDeposit() / CLHEP::MeV
      //        << " (MeV) "
      //        << G4endl;
   }
 
   /* Get the target physical volume and compare it against the current physical volume */
   if (current_physicalVolume == target_physicalVolume)
   {
      G4double total_energyDeposite = current_step->GetTotalEnergyDeposit() / CLHEP::MeV;
      if ( total_energyDeposite > 0.0 )
      {
         // G4cout << "    --> Step Action done in: " 
         //       << current_physicalVolume->GetName()
         //       << " with Energy Deposite "   
         //       << total_energyDeposite
         //       << " (MeV) "
         //       << G4endl;

         (this->fevent_action)->add_energyDepositePerEvent( total_energyDeposite );
      }

      /* Get the current stacking track for the current step*/
      // G4Track * current_stack = current_step->GetTrack();
   }

}

}