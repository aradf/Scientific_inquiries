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
#ifndef DetectorMessenger_h
#define DetectorMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

class CDetectorConstruction;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;

/*
 * - Create user-defined UI command to manipulate some of the properties of application object 
     (e.g. write a command to be able to setTargetMaterial in YourDetectorConstruction)
 * - Write user-defined Messenger class (with defining/adding your UIcommand-s) to your object
     and add user-defined target object (e.g. YourDetectorConstruction) which is responsible for
     instantiating and deleting the messenger object
 * - Messenger:
 * - Handles UIcommand-s targeting a given object (e.g. YourDetectorConstruction) and
     derived from the G4UImesenger base class (e.g. class YourDetectorMessenger :
     public G4UImessenger { … )
 * - UIcommand-s (UIdirectory(-is)) are created.
*/
class CDetectorMessenger: public G4UImessenger
{
  public:
  
    CDetectorMessenger(CDetectorConstruction * );
   ~CDetectorMessenger();
    
    virtual void SetNewValue(G4UIcommand*, G4String);
    
  private:
  
    CDetectorConstruction*      fdetector_constructor;
    
    G4UIdirectory*             ftest_electromagneticDir;
    G4UIdirectory*             fdetector_directory;    
    G4UIcmdWithAString*        fmaterial_command;
    G4UIcmdWithADoubleAndUnit* fsize_command;
};
#endif

