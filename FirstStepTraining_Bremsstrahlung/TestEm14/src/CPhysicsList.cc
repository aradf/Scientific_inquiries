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

#include "CPhysicsList.hh"
#include "CPhysicsListMessenger.hh"
 
#include "CPhysListEmStandard.hh"
#include "PhysListEmLivermore.hh"
#include "PhysListEmPenelope.hh"

#include "G4LossTableManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

/*
 * particles
 */

#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BosonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"

CPhysicsList::CPhysicsList() : G4VModularPhysicsList(),
                               felectromagnetic_physicsList(nullptr),
                               fmessenger(nullptr)
{
  G4LossTableManager::Instance();  
  fmessenger = new CPhysicsListMessenger(this);
  SetVerboseLevel(1);

  /*
   * EM physics
   */
  felectromagnetic_name = G4String( "standard" );
  felectromagnetic_physicsList = new CPhysListEmStandard( felectromagnetic_name );

  /*
   *  Add new units for Cross Sections.
   */
  new G4UnitDefinition( "mm2/g", 
                        "mm2/g",
                        "Surface/Mass", 
                        mm2 / g);

  new G4UnitDefinition( "um2/mg", 
                        "um2/mg",
                        "Surface/Mass", 
                        um * um / mg);
}

CPhysicsList::~CPhysicsList()
{
  delete fmessenger;
}

void CPhysicsList::ConstructParticle()
{
    G4BosonConstructor  pboson_constructor;
    pboson_constructor.ConstructParticle();

    G4LeptonConstructor plepton_constructor;
    plepton_constructor.ConstructParticle();

    G4MesonConstructor pmeson_constructor;
    pmeson_constructor.ConstructParticle();

    G4BaryonConstructor pbaryon_constructor;
    pbaryon_constructor.ConstructParticle();

    G4IonConstructor pion_constructor;
    pion_constructor.ConstructParticle();

    G4ShortLivedConstructor pshort_livedConstructor;
    pshort_livedConstructor.ConstructParticle();  
}

void CPhysicsList::ConstructProcess()
{

  /*
   *  Transportation
   */
  AddTransportation();

  /*
   *  Electromagnetic physics list
   */
  felectromagnetic_physicsList->ConstructProcess();
  
  /*
   *  Em options
   */
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetIntegral( false );
}

void CPhysicsList::add_physicslist(const G4String& name)
{
  if ( verboseLevel > 0 ) 
  {
      G4cout << "PhysicsList::add_physicslist: <" 
             << name 
             << ">" 
             << G4endl;
  }
  
  if ( name == felectromagnetic_name ) 
      return;

  if ( name == "standard" ) 
  {
      felectromagnetic_name = name;
      delete felectromagnetic_physicsList;
      felectromagnetic_physicsList = new CPhysListEmStandard( name );
   
  } 
  else if ( name == "livermore" ) 
  {
      felectromagnetic_name = name;
      delete felectromagnetic_physicsList;
      felectromagnetic_physicsList = new PhysListEmLivermore( name );
    
  } 
  else if ( name == "penelope" ) 
  {
      felectromagnetic_name = name;
      delete felectromagnetic_physicsList;
      felectromagnetic_physicsList = new PhysListEmPenelope( name );
                    
  } 
  else 
  {
      G4cout << "PhysicsList::add_physicslist: <" << name << ">"
            << " is not defined"
            << G4endl;
  }
}


void CPhysicsList::SetCuts()
{ 
  /*
   * fixed lower limit for cut
   */
  G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(100 * eV, 
                                                                  1 * GeV);

  /*
   * Call base class method to set cuts which default value can be
   * modified via /run/setCut/* commands
   */
  G4VUserPhysicsList::SetCuts();

  DumpCutValuesTable();
}

