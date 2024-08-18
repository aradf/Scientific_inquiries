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

#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "globals.hh"

class G4Event;
class CDetectorConstruction;

class CPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
   CPrimaryGeneratorAction( CDetectorConstruction * );    
   ~CPrimaryGeneratorAction();

public:
   /*
    * The GeneratePrimaries interface method (pure virtual) is invoked
    * by the G4RunManager during the event-loop (in its G4Runmanager::GenerateEvent
    * method).  This method describes how the primary particle(s) are produced in
    * an event.
    */
   virtual void GeneratePrimaries(G4Event*);
   
   G4ParticleGun * get_particleGun() 
   {
      return fparticle_gun;
   };

private:
   G4ParticleGun*        fparticle_gun;
   CDetectorConstruction* fdetector_constructor;
};


#endif


