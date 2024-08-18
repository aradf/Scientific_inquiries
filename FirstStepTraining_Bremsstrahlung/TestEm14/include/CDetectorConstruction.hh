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

#ifndef DetectorConstruction_h
#define DetectorConstruction_h

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4LogicalVolume;
class G4Material;
class CDetectorMessenger;

/*
 * CDetectorConstruction describes the geometrical setup, including volume and their
 * shape, position adn material definition. 
 */
class CDetectorConstruction : public G4VUserDetectorConstruction
{
public:
   CDetectorConstruction();
   virtual ~CDetectorConstruction();

public:
   /* 
    * The Construct method is pure virutal and it is invoked by G4RunManager at initialization.
    * Create all materials used in the geometry.
    * Describe the detector geometry by creating and positioning volumes.
    * Return the pointer to the root (world) of your geometry heirarchy.
    */
   virtual G4VPhysicalVolume* Construct();

   /*
    * The set_boxSize is invoked by the Detector messanger class to change the box size
    * from one run to another.
    */
   void set_boxSize ( G4double );              
   void set_material ( G4String );            

public:
   const
   G4VPhysicalVolume* get_world()      
   { 
      return fphysical_VolumeBox; 
   };           
   
   G4double get_size()       
   {
      return fboxsize;
   };      

   G4Material* get_material()   
   {
      return fmaterial;
   };
   
   void print_parameters();
                       
private:
   G4VPhysicalVolume*    fphysical_VolumeBox;
   G4LogicalVolume*      flogical_VolumeBox;
     
   G4double              fboxsize;
   G4Material*           fmaterial;     
     
   CDetectorMessenger* fdetector_messenger;

private:
   void               define_materials();
   G4VPhysicalVolume* construct_volumes();     
};

#endif

