//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
#ifndef Run_h
#define Run_h

#include "G4Run.hh"
#include "globals.hh"
#include <map>

class CDetectorConstruction;
class G4ParticleDefinition;

/* 
 * Optional User Actions:
 * - G4Run is a collection of G4Event(s).  G4EVent is a collection of G4Track(s).
 * - During a single Run, events are taken and processed one by one in an event loop.
 * - At the start of a run (G4RunManager::BeamOn()), The geometry is optimised for tracking.  * 
 * - (voxelization), physics tables are built, then event processing starts i.e. entering into the event-loop.
 * - As long as event processing is running, i.e. during the run, the user cannot modify neither the geometry
 *   (i.e. the detector setup) nor the physics settings.
 *   They can be changed thought between run(s) but the G4RunManager needs to be informed.
 *   (re-optimised or re-construct) geometry, re-build physics tables: if the geometry has been changed, 
 *   depending on the modification: GeometryHasBeenModified() re-voxelizatin but no re-construct, 
 *   ReinitializeGeometry() complete re-construct, or with UI commands /run/geometryModified or /run/reinitializeGeometry
 *   Same for the physics: PhysicsHasBeenModified() or /run/physicsModified 
 */
class CRun : public G4Run
{
  public:
    CRun(CDetectorConstruction*);
   ~CRun();

  public:
    virtual void Merge(const G4Run*);
    void set_primary(G4ParticleDefinition* particle, G4double energy);
    void count_processes(G4String process_name);
    void sum_tracks (G4double track); 
    void sum_energyTransfered (G4double energy);            
    void end_ofSingleRun();

  private:
    CDetectorConstruction*  fdetector_constructor;
    G4ParticleDefinition*  fparticle_definition;
    G4double  fkinetic_energy;

    /*
     * The standard template library map is a stl container that holds a pairs of key-values.
     * In this case it holds pairs of G4String and G4int key values.
     */
    std::map< G4String, G4int >    fproccess_counterMap;
    G4int    ftotal_count;   //all processes counter
    G4double fsum_track;     //sum of trackLength
    G4double fsum_trackSquared;    //sum of trackLength*trackLength
    G4double fenergy_transfered;   //energy transfered to charged secondaries
};

#endif

