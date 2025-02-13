
#include <CLab04SensitiveDetector.hh>
// #include "CLab04DetectorConstruction.hh"

#include "G4RunManager.hh"
#include "G4AnalysisManager.hh"
#include "G4SystemOfUnits.hh"


namespace NUCE427LAB04
{


CLab04SensitiveDetector::CLab04SensitiveDetector(G4String detector_name) : G4VSensitiveDetector( detector_name )
{
    G4cout << "INFO: CLab04SensitiveDetector Constructor ..."
           << G4endl;



}

CLab04SensitiveDetector::~CLab04SensitiveDetector() 
{


    
}

G4bool CLab04SensitiveDetector::ProcessHits(G4Step * current_step , G4TouchableHistory * ROHist)
{
    G4Track* current_track = current_step->GetTrack();
    current_track->SetTrackStatus( fStopAndKill );
    
    // G4LogicalVolume* current_logicalVolume = current_step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
    // const CLab04DetectorConstructor* detector_construction = static_cast< const CLab04DetectorConstructor* > (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    // G4LogicalVolume* scoring_logicalVolume = detector_construction->getscoring_logicalVolume();
    // if( current_logicalVolume != scoring_logicalVolume )
    //    return false;


    G4StepPoint* prestep_point = current_step->GetPreStepPoint();
    G4StepPoint* poststep_point = current_step->GetPostStepPoint();

    G4ThreeVector position_photon = prestep_point->GetPosition();
    G4ThreeVector momentum_photon = prestep_point->GetMomentum();

    G4double wlen = (1.239841939 * CLHEP::eV / momentum_photon.mag() ) * 1E+03;

    #warning "Add to calculate the photon position shen encountered the NaI(IT) scintillator ..."
    G4cout << "Photon Position " 
           << position_photon
           << G4endl;

    const G4VTouchable* touchable = current_step->GetPreStepPoint()->GetTouchable();
//     G4int copy_number = touchable->GetCopyNumber();
//     G4cout << "Copy Number"
//            << copy_number
//            << G4endl;

    G4VPhysicalVolume * scintillator_physicalVolume = touchable->GetVolume();
    G4ThreeVector position_detector = scintillator_physicalVolume->GetTranslation();

    // G4cout << "position_detector "
    //        << position_detector
    //        << G4endl;

    G4AnalysisManager* manager_analysis = G4AnalysisManager::Instance();

    G4int event_number = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    manager_analysis->FillNtupleIColumn(0, 0, event_number);
    manager_analysis->FillNtupleDColumn(0, 1, position_photon[0]);
    manager_analysis->FillNtupleDColumn(0, 2, position_photon[1]);
    manager_analysis->FillNtupleDColumn(0, 3, position_photon[2]);
    manager_analysis->FillNtupleDColumn(0, 4, wlen);

    manager_analysis->AddNtupleRow(0);

    return true;
}

}