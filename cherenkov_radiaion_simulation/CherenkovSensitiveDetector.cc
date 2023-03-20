#include "CherenkovSensitiveDetector.hh"

CCherenkovSensitiveDetector::CCherenkovSensitiveDetector(G4String detector_name) : G4VSensitiveDetector(detector_name)
{


}

CCherenkovSensitiveDetector::~CCherenkovSensitiveDetector()
{


}

/**  
 * This method is invoked by G4SteppingManager when a step is composed in the G4LogicalVolume.  
 * The first argument of this method is G4Step object of the current step.  The second argument is 
 * obsolete.  In this method, one or more G4VHit objects are constructed when the current step has
 * information meaningful to the detector.
*/
G4bool CCherenkovSensitiveDetector::ProcessHits(G4Step * particle_step , G4TouchableHistory * ROHist)
{
    /**
     * Get particle track which entered the sensitive volume.
     * A hit in Geant4 is the snaphost of the physical interaction of a
     * track in the sensitive region of a detector.   This object can 
     * store information (members) associated with a G4Step object.  These 
     * member(s) could be:
     * - the position and time of the step,
     * - the momentum and energy of the track,
     * - the energy deposition of the step,
     * - the geometrical information,
     */
    G4Track * particle_track = particle_step->GetTrack();
    
    particle_track->SetTrackStatus(fStopAndKill);

    /**
     * When a particle enteres the sensitive volume, it has a beginning and an end point.
     * This is the point where the photon enteres the sensitive detector.
     */
    G4StepPoint * prestep_point = particle_step->GetPreStepPoint();

    /**
     * When a particle enteres the sensitive volume, it has a beginning and an end point.
     * This is the point where the photon leaves the sensitive detector.

     * Information member(s) in G4StepPoint (PreStepPoint and PostStepPoint) includes:
     * - (x, y, z, t)
     * - (px, py, pz, Ek)
     * - Momentum direction (unit vector)
     * - Pointers to physical volumes
     * - Beta, gamma
     * - Polarization
     * - Step status
     * - Total track length
     * - Global time (time since the current event began)
     * - Local time (time since the current track began)
     * - Proper time
     */
    G4StepPoint * poststep_point = particle_step->GetPostStepPoint();
    G4ThreeVector position_photon = prestep_point->GetPosition();
    G4ThreeVector momentum_photon = prestep_point->GetMomentum();
    G4double wave_lenth = (1.239841939 * eV/momentum_photon.mag())*1E+03;

    // G4cout << "Photon Position: " << position_photon << G4endl;

    const G4VTouchable * touchable = particle_step->GetPreStepPoint()->GetTouchable();
    G4int copy_number = touchable->GetCopyNumber();

    // G4cout << "Copy Number: " << copy_number << G4endl;

    G4VPhysicalVolume * physical_volume = touchable->GetVolume();
    G4ThreeVector position_detector = physical_volume->GetTranslation();
   
    G4cout << "Detector position: " << position_detector << G4endl;
    G4int event_number = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    G4AnalysisManager * manager_analysis = G4AnalysisManager::Instance();

    manager_analysis->FillNtupleIColumn(0, 0, event_number);
    manager_analysis->FillNtupleDColumn(0, 1, position_photon[0]);
    manager_analysis->FillNtupleDColumn(0, 2, position_photon[1]);
    manager_analysis->FillNtupleDColumn(0, 3, position_photon[2]);
    manager_analysis->FillNtupleDColumn(0, 4, wave_lenth);
    manager_analysis->AddNtupleRow(0);


    manager_analysis->FillNtupleIColumn(1, 0, event_number);
    manager_analysis->FillNtupleDColumn(1, 1, position_detector[0]);
    manager_analysis->FillNtupleDColumn(1, 2, position_detector[1]);
    manager_analysis->FillNtupleDColumn(1, 3, position_detector[2]);
    manager_analysis->AddNtupleRow(1);

    return true;    
}
