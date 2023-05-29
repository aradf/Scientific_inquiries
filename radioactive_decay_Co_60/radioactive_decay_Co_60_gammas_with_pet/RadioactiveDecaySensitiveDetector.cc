
#include "RadioactiveDecaySensitiveDetector.hh"

CRadioactiveDecaySensitiveDetector::CRadioactiveDecaySensitiveDetector(G4String sensitivedetector_name ) : G4VSensitiveDetector(sensitivedetector_name)
{

}

CRadioactiveDecaySensitiveDetector::~CRadioactiveDecaySensitiveDetector()
{

}

/**
 * ProcessHits method is invoked by the G4SteppingManager when a step is composed in the G4LogicalVolume which has the 
 * pointer to the this sensitive detector.  The first input is a pointer variable of type current step.  In this method,
 * one or mor G4VHit objects are to be constructed, if the current step has meaning (related information).  The Initialize
 * method is invoked at the beginning of each event.  The argument of this method is an object of type G4HCofThisEvent where
 * the hit produceds in this particular event are stored.  EndofEvent() method is invoked at the end of each event.   
 */
G4bool CRadioactiveDecaySensitiveDetector::ProcessHits(G4Step * current_step, G4TouchableHistory *)
{

   G4Track *track = current_step->GetTrack();
   
   track->SetTrackStatus(fStopAndKill);

   G4StepPoint * prestep_point = current_step->GetPreStepPoint();
   G4StepPoint * poststep_point = current_step->GetPostStepPoint();

   G4ThreeVector position_photon = prestep_point->GetPosition();
   G4ThreeVector momentum_photon = prestep_point->GetMomentum();

   //G4double time = prestep_point->GetGlobalTime();
   /* Formula E = h * f
    * E is Energy, h is the planks constant , f is the frequency.
    */
   G4double wlen = (1.239841939 * eV/momentum_photon.mag())*1E+03;


   // G4cout << "Photon Position: " << position_photon << G4endl;

   const G4VTouchable * touchable = current_step->GetPreStepPoint()->GetTouchable();
   G4int copy_number = touchable->GetCopyNumber();
   
   // G4cout << "Copy number: " << copy_number << G4endl;

   G4VPhysicalVolume * physical_volume = touchable->GetVolume();
   G4ThreeVector position_detector = physical_volume->GetTranslation();
   
   G4cout << "Detector position: " << position_detector << G4endl;
   G4cout << "Photon Wavelength: " << wlen << G4endl;

   G4int event_number = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

   G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();

   manager_analysis->FillNtupleIColumn(0, 0, event_number);
   manager_analysis->FillNtupleDColumn(0, 1, position_photon[0]);
   manager_analysis->FillNtupleDColumn(0, 2, position_photon[1]);
   manager_analysis->FillNtupleDColumn(0, 3, position_photon[2]);
   manager_analysis->FillNtupleDColumn(0, 4, wlen);
   //manager_analysis->FillNtupleDColumn(0, 5, time);
   manager_analysis->AddNtupleRow(0);

   manager_analysis->FillNtupleIColumn(1, 0, event_number);
   manager_analysis->FillNtupleDColumn(1, 1, position_detector[0]);
   manager_analysis->FillNtupleDColumn(1, 2, position_detector[1]);
   manager_analysis->FillNtupleDColumn(1, 3, position_detector[2]);
   manager_analysis->AddNtupleRow(1);


   return true;
}
