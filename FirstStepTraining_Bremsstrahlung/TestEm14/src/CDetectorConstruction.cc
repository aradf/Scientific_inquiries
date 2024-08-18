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
#include "CDetectorConstruction.hh"
#include "CDetectorMessenger.hh"

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"

#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4RunManager.hh"


CDetectorConstruction::CDetectorConstruction() : G4VUserDetectorConstruction(),
                                               fphysical_VolumeBox(0), 
                                               flogical_VolumeBox(0), 
                                               fmaterial(0),
                                               fdetector_messenger(0)
{
   fboxsize = 100 * m;
   define_materials();
   set_material( "Water" );  
   fdetector_messenger = new CDetectorMessenger( this );
}

CDetectorConstruction::~CDetectorConstruction()
{ 
   delete fdetector_messenger;
}

G4VPhysicalVolume* CDetectorConstruction::Construct()
{
  return construct_volumes();
}

void CDetectorConstruction::define_materials()
{
  /*
   * define elements
   */
  G4double z = 0.0 , a = 0.0 ;
  
  G4Element* H  = new G4Element("Hydrogen" ,"H" , z= 1., a=   1.01*g/mole);
  G4Element* N  = new G4Element("Nitrogen" ,"N" , z= 7., a=  14.01*g/mole);
  G4Element* O  = new G4Element("Oxygen"   ,"O" , z= 8., a=  16.00*g/mole);
  G4Element* Na = new G4Element("Sodium"   ,"Na", z=11., a=  22.99*g/mole);
  G4Element* Ge = new G4Element("Germanium","Ge", z=32., a=  72.59*g/mole);
  G4Element* I  = new G4Element("Iodine"   ,"I" , z=53., a= 126.90*g/mole);
  G4Element* Bi = new G4Element("Bismuth"  ,"Bi", z=83., a= 208.98*g/mole);
  
  /*
   * define materials
   */

  G4double density = 0.0;
  G4int ncomponents = 0 , natoms = 0;
  G4double fractionmass = 0.0;  
  
  G4Material* Air = 
  new G4Material("Air", density= 1.290*mg/cm3, ncomponents=2);
  Air->AddElement(N, fractionmass=70.*perCent);
  Air->AddElement(O, fractionmass=30.*perCent);

  G4Material* H2l = 
  new G4Material("H2liquid", density= 70.8*mg/cm3, ncomponents=1);
  H2l->AddElement(H, fractionmass=1.);

  G4Material* H2O = 
  new G4Material("Water", density= 1.000*g/cm3, ncomponents=2);
  H2O->AddElement(H, natoms=2);
  H2O->AddElement(O, natoms=1);
  H2O->SetChemicalFormula("H_2O");
  H2O->GetIonisation()->SetMeanExcitationEnergy(78.0*eV);

  new G4Material("liquidArgon", z=18., a= 39.95*g/mole, density= 1.390*g/cm3);
  
  new G4Material("Carbon"     , z=6.,  a= 12.01*g/mole, density= 2.267*g/cm3);

  new G4Material("Aluminium"  , z=13., a= 26.98*g/mole, density= 2.700*g/cm3);

  new G4Material("Silicon"    , z=14., a= 28.09*g/mole, density= 2.330*g/cm3);
  
  new G4Material("Chromium"   , z=24., a= 51.99*g/mole, density= 7.140*g/cm3);
      
  new G4Material("Copper"     , z=29., a= 63.55*g/mole, density= 8.920*g/cm3);  

  new G4Material("Germanium"  , z=32., a= 72.61*g/mole, density= 5.323*g/cm3);

  new G4Material("Nb"         , z=41., a= 92.906*g/mole,density= 8.57*g/cm3);
    
  G4Material* NaI = 
  new G4Material("NaI", density= 3.67*g/cm3, ncomponents=2);
  NaI->AddElement(Na, natoms=1);
  NaI->AddElement(I , natoms=1);
  NaI->GetIonisation()->SetMeanExcitationEnergy(452*eV);
  
  G4Material* Iod = 
  new G4Material("Iodine", density= 4.93*g/cm3, ncomponents=1);
  Iod->AddElement(I , natoms=1);
  
  G4Material* BGO = 
  new G4Material("BGO", density= 7.10*g/cm3, ncomponents=3);
  BGO->AddElement(O , natoms=12);
  BGO->AddElement(Ge, natoms= 3);
  BGO->AddElement(Bi, natoms= 4);  

  new G4Material("Iron"       , z=26., a= 55.85*g/mole, density= 7.870*g/cm3);

  new G4Material("Tungsten"   , z=74., a=183.85*g/mole, density= 19.25*g/cm3);
  
  new G4Material("Gold"       , z=79., a=196.97*g/mole, density= 19.30*g/cm3);  

  new G4Material("Lead"       , z=82., a=207.19*g/mole, density= 11.35*g/cm3);

  new G4Material("Uranium"    , z=92., a=238.03*g/mole, density= 18.95*g/cm3);

  /// G4cout << *(G4Material::get_material()) << G4endl;
}

G4VPhysicalVolume* CDetectorConstruction::construct_volumes()
{
  /*
   * Cleanup old geometry
   */
  G4GeometryManager::GetInstance()->OpenGeometry();
  G4PhysicalVolumeStore::GetInstance()->Clean();
  G4LogicalVolumeStore::GetInstance()->Clean();
  G4SolidStore::GetInstance()->Clean();

  /*
   *  A simple box (shape) as the detector/target filled with with water as material
   *  placed in a box (shape) “world” volume.
   */
  G4Box * solid_box = new G4Box("Container",   
                                fboxsize/2, 
                                fboxsize/2, 
                                fboxsize/2 );  

  /*
   * The shape and dimensions of the volume i.e. a G4VSolid
   * -  The material of the volume i.e. G4Material that is the minimally required additional
   *    information beyond the solid
   * -  Additional, optional information such as magnetic field (G4FieldManager) or user
   *    defined limits (G4UserLimits), etc.
   */
  flogical_VolumeBox = new G4LogicalVolume(solid_box,                //its shape
                                           fmaterial,                //its material
                                           fmaterial->GetName());    //its name

  /* –  Placement volume: one positioned volume, i.e. one G4VPhysicalVolume object
   *    represents one “real” volume
   * –  Repeated volume: one volume positioned many times, i.e. one
   *    G4VPhysicalVolume object represents multiple copies of “real” volumes
   *    (reduces memory by exploiting symmetry)
   * –  Replica volumes: the multiple copies of the volume are all identical
   *    Parameterised volumes: the multiple copies of a volume can be different in
   *    size, solid type, or material that can all be parameterised as a function of the
   *    copy number
   */ 
  fphysical_VolumeBox = new G4PVPlacement(0,                         //no rotation
                                          G4ThreeVector(),           //at (0,0,0)
                                          flogical_VolumeBox,        //its logical volume
                                          fmaterial->GetName(),      //its name
                                          0,                         //its mother  volume
                                          false,                     //no boolean operation
                                          0);                        //copy number
                           
  print_parameters();
 
  /*
   * Always return the root physical volume.
   */
  return fphysical_VolumeBox;
}

void CDetectorConstruction::print_parameters()
{
  G4cout << "\n The Box is " 
         << G4BestUnit( fboxsize , "Length")
         << " of " 
         << fmaterial->GetName() 
         << G4endl;

  G4cout << "\n" 
         << fmaterial 
         << G4endl;	 
}

void CDetectorConstruction::set_material(G4String material_choice)
{
  /*
   * search the material by its name, or build it from nist data base
   */
  G4Material* ptto_material = G4NistManager::Instance()->FindOrBuildMaterial(material_choice);
  
  if (ptto_material != nullptr) 
  {
      fmaterial = ptto_material;
      G4RunManager::GetRunManager()->PhysicsHasBeenModified();
  } 
  else 
  {
      G4cout << "\n--> warning from DetectorConstruction::set_material : "
            << material_choice 
            << " not found" 
            << G4endl;  
  } 
}

void CDetectorConstruction::set_boxSize(G4double value)
{
  fboxsize = value;
  G4RunManager::GetRunManager()->ReinitializeGeometry();
}

