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

#include "CPhysicsListMessenger.hh"
#include "CPhysicsList.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"

CPhysicsListMessenger::CPhysicsListMessenger(CPhysicsList* some_physics)
                                             :G4UImessenger(),
                                             fphysics_list(some_physics),
                                             fphysics_dir(nullptr), 
                                             flist_command(nullptr)
{ 
  fphysics_dir = new G4UIdirectory( "/testem/phys/" );
  fphysics_dir->SetGuidance( "physics list commands" );

  flist_command = new G4UIcmdWithAString( "/testem/phys/addPhysics" ,
                                     this );  
  flist_command->SetGuidance( "Add modular physics list." );
  flist_command->SetParameterName( "PList" ,
                               false);
  flist_command->AvailableForStates( G4State_PreInit );  
}

CPhysicsListMessenger::~CPhysicsListMessenger()
{
  delete flist_command;
  delete fphysics_dir;
}

void CPhysicsListMessenger::SetNewValue(G4UIcommand* some_command,
                                          G4String new_value)
{       
  
  if( some_command == flist_command )
  { 
     G4cout << new_value << G4endl; 
     fphysics_list->add_physicslist(new_value);
  }
}

