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

#ifndef PhysicsList_h
#define PhysicsList_h 

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

class CPhysicsListMessenger;
class G4VPhysicsConstructor;

/*
 * Physics List is an object that is responsible to:
 * - Specify all the particles that will be used in the simulation application.  
 * - Togheter with the list of physics process assigned to them.
 * 
 * The G4VModularPhysicList interface is provided by Geant4 toolkit to desribe the physics setup, including 
 * definition of all particles and thier physics interactions, process.
 * Its ConstructParticle() and ConstructProcess() interface methods (pure virtual) are invoked 
 * by the G4RunManager (G4RunManagerKernel and process construction is invoked indirectly) at initialisation.
 */
class CPhysicsList: public G4VModularPhysicsList
{
public:
    CPhysicsList();
   ~CPhysicsList();

    virtual void ConstructParticle();
    virtual void ConstructProcess();
    virtual void SetCuts();
    void add_physicslist(const G4String& name);
             
private:    
    G4VPhysicsConstructor * felectromagnetic_physicsList; 
    G4String felectromagnetic_name;
    
    CPhysicsListMessenger *  fmessenger;         
};

#endif

