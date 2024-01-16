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

#ifndef NUCE427LAB02EventAction_h
#define NUCE427LAB02EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

// Event action class

namespace NUCE427LAB02
{

class RunAction;



/*
  An event is the basic unit of simulation in Geant4.  In the beginning of processing, primary tracks are generated.
  These primary tracks are pushed to stack.  A single track is poped up from the stack one by one and 'tracked'.
  A run is a collection of events.  Every time the '/run/beamOn' is invoked an instance of RunAction class is
  invoked.  There are many instances of EventAction class present.  If you want to know, how many particles
  reached your detector, you should investigate every event.  One event will have many particles reached the 
  detector.  I will sum up the total count number from each event in a RunAction and print the results in
  EndOfRunAction method.
*/


class EventAction : public G4UserEventAction
{
public:
  EventAction(RunAction* run_action);
  ~EventAction();

  virtual void BeginOfEventAction(const G4Event* event);
  virtual void EndOfEventAction(const G4Event* event);
  void add_energy_deposite(G4double energy) {
    fenergy_deposite = fenergy_deposite + energy; 
  }

private:
  RunAction* current_runaction = nullptr;
  G4double fenergy_deposite;
};

}

#endif


