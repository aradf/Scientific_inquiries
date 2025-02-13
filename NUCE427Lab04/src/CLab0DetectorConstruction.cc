#include "CLab04DetectorConstruction.hh"
#include "CLab04SensitiveDetector.hh"

#include <G4VPhysicalVolume.hh>
#include <G4NistManager.hh>
#include <G4Box.hh>
#include <G4Tubs.hh>
#include <G4SystemOfUnits.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4Material.hh>
#include <G4VisAttributes.hh>

#warning "Added to simulate a Uniform Electric Field in Geant4 ..."
/*
#include <G4ElectricField.hh>
#include <G4UniformElectricField.hh>
#include <G4FieldManager.hh>
#include "G4FieldBuilder.hh"
*/

namespace NUCE427LAB04
{

CLab04DetectorConstructor::CLab04DetectorConstructor() : G4VUserDetectorConstruction()
{
    G4cout << "INFO: CLab04DetectorConstructor Constructor ..."
           << G4endl;

    world_material = nullptr;

    world_solid = nullptr;
    scintillator_solid = nullptr;

    world_logialVolume = nullptr;
    scintillator_logicalVolume = nullptr;

    world_physicalVolume = nullptr;
    scintillator_physicalVolume = nullptr;

    xworld = 0.5 * CLHEP::m;
    yworld = 0.5 * CLHEP::m;
    zworld = 0.5 * CLHEP::m;

    define_materials();
}

CLab04DetectorConstructor::~CLab04DetectorConstructor()
{


}

void CLab04DetectorConstructor::define_materials()
{
    G4NistManager* nist_manager = G4NistManager::Instance();

    NaI_material = new G4Material("NaI", 3.67 * CLHEP::g / CLHEP::cm3, 2);
    Na_element = nist_manager->FindOrBuildElement("Na");
    I_element  = nist_manager->FindOrBuildElement("I");
    NaI_material->AddElement(Na_element, 1);
    NaI_material->AddElement(I_element, 1);
}

G4VPhysicalVolume* CLab04DetectorConstructor::Construct()
{
    G4cout << "INFO: CLab04DetectorConstructor Construct ..."
           << G4endl;

    /*
     * World material definition.
     */
    G4NistManager* nist_manager = G4NistManager::Instance();
    world_material = nist_manager->FindOrBuildMaterial("G4_AIR");


    /*
     * World Solid definition.
     */
    world_solid = new G4Box( "solidWorld", 
                             xworld, 
                             yworld, 
                             zworld );

    /*
     * World logical Volume definition.
     */
    world_logialVolume = new G4LogicalVolume( world_solid,
                                              world_material,
                                              "logicWorld" );

    /*
     * World physical volume defintion.
     */
    world_physicalVolume = new G4PVPlacement( 0,
                                              G4ThreeVector( 0.0, 0.0, 0.0),
                                              world_logialVolume,
                                              "physicalWorld",
                                              0,
                                              false,
                                              0,
                                              true);

    construct_scintillator();

    fscoring_logicalVolume = scintillator_logicalVolume;
   
    return world_physicalVolume;
}



void CLab04DetectorConstructor::ConstructSDandField()
{

#warning "Added to simulate a Uniform Electric Field in Geant4 ..."
/*
    G4ElectricField* electrical_field = new G4UniformElectricField( G4ThreeVector( 0.0, 
                                                                                   6.0E2* CLHEP::kilovolt / CLHEP::cm, 
                                                                                   0.0 ) );
    // Set field to the field builder
    auto fieldBuilder = G4FieldBuilder::Instance();
    fieldBuilder->SetGlobalField( electrical_field );

    // Construct all Geant4 field objects
    fieldBuilder->SetFieldType(kElectroMagnetic);
    fieldBuilder->ConstructFieldSetup();
 */

    CLab04SensitiveDetector* sensitive_detector = new CLab04SensitiveDetector( "sensitive_detector" );
    scintillator_logicalVolume->SetSensitiveDetector( sensitive_detector );

}

void CLab04DetectorConstructor::construct_scintillator()
{
    scintillator_solid = new G4Tubs( "solidScintillator",
                                      10 * CLHEP::cm,
                                      20 * CLHEP::cm,
                                      30 * CLHEP::cm,
                                      0 * CLHEP::deg, 
                                      360 * CLHEP::deg );

    scintillator_logicalVolume = new G4LogicalVolume( scintillator_solid,
                                                      NaI_material,
                                                     "logicalScintillator");
    scintillator_physicalVolume = new G4PVPlacement( 0,
                                                     G4ThreeVector( 0.0, 0.0, 0.0),
                                                     scintillator_logicalVolume,
                                                     "physicalScintillator",
                                                     world_logialVolume,
                                                     false,
                                                     0,
                                                     true);
    
    green = new G4VisAttributes( G4Colour::Green() );
    scintillator_logicalVolume->SetVisAttributes( green );

}

}