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

#include "ActionInitialization.hh"
#include "RunAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "StackingAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"

using namespace NUCE427LAB02;

namespace NUCE427LAB02
{

ActionInitialization::ActionInitialization()
{}

ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::BuildForMaster() const
{
  /*
  SetUserAction(new RunAction);
  */
}

/*
 A running action is a class abstracting away the features of /run/initializtion.
 A run has many events represented by a class Event Action.
 A an Event has many steps represented by stepping Action.
*/

void ActionInitialization::Build() const
{
  PrimaryGeneratorAction * primary_generator_action = new PrimaryGeneratorAction();
  SetUserAction(primary_generator_action); 

  RunAction * run_action = new RunAction();
  SetUserAction(run_action);

  EventAction * event_action = new EventAction(run_action);
  SetUserAction(event_action);

  SteppingAction * stepping_action = new SteppingAction(event_action);
  SetUserAction(stepping_action);

  // Remove for production
  StackingAction * stacking_action = new StackingAction();
  SetUserAction(stacking_action);

}

}

