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

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "CDetectorMessenger.hh"
#include "CDetectorConstruction.hh"

CDetectorMessenger::CDetectorMessenger(CDetectorConstruction * detector_constructor)
                                       :G4UImessenger(),
                                       fdetector_constructor(detector_constructor),
                                       ftest_electromagneticDir(nullptr),
                                       fdetector_directory(nullptr),    
                                       fmaterial_command(nullptr),
                                       fsize_command(nullptr)
{ 
  ftest_electromagneticDir = new G4UIdirectory("/testem/");
  ftest_electromagneticDir->SetGuidance("commands specific to this example");
  
  fdetector_directory = new G4UIdirectory("/testem/det/");
  fdetector_directory->SetGuidance("detector construction");
  
  fmaterial_command = new G4UIcmdWithAString("/testem/det/set_material",this);
  fmaterial_command->SetGuidance("Select material of the box.");
  fmaterial_command->SetParameterName("choice",false);
  fmaterial_command->AvailableForStates(G4State_PreInit,G4State_Idle);
  fmaterial_command->SetToBeBroadcasted(false);
  
  fsize_command = new G4UIcmdWithADoubleAndUnit("/testem/det/set_boxSize",this);
  fsize_command->SetGuidance("Set size of the box");
  fsize_command->SetParameterName("Size",false);
  fsize_command->SetRange("Size>0.");
  fsize_command->SetUnitCategory("Length");
  fsize_command->AvailableForStates(G4State_PreInit,G4State_Idle);
  fsize_command->SetToBeBroadcasted(false);
}

CDetectorMessenger::~CDetectorMessenger()
{
  delete fmaterial_command;
  delete fsize_command; 
  delete fdetector_directory;  
  delete ftest_electromagneticDir;
}

/*
 * SetNewValue(G4UIcommand* cmd, G4String newValue) is invoked when a command is invoked:
 * find out which one of your commands has triggered this call by comparing your command
   pointers (stored as members of your Messenger class) to the cmd parameter
 * Convert the newValue command parameter string to the appropriate value
 * Apply the corresponding changes to your target class
 */
void CDetectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{ 
  if( command == fmaterial_command )
  { 
     fdetector_constructor->set_material(newValue);
  }
   
  if( command == fsize_command )
  { 
     fdetector_constructor->set_boxSize(fsize_command->GetNewDoubleValue(newValue));
  }
}


