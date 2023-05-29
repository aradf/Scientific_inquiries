#ifndef RADIOACTIVE_DECAY_GENERATOR_HH
#define RADIOACTIVE_DECAY_GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4Geantino.hh"
#include "G4IonTable.hh"

/**
 * Class description.
 * This class creates an instance of a primary particle generator where
 * the initial state of the primary event is defined.  It is a mandatory 
 * action class for primary vertex/particle generation. 
 * This class should 
 *  - have one or more G4VPrimaryGenerator classes such as G4ParticleGun 
 *  - set/change properties of primary generator(s)
 */
class CRadioactiveDecayPrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
   CRadioactiveDecayPrimaryGenerator();
   ~CRadioactiveDecayPrimaryGenerator();

   /**
    * This virtual method is invoked from G4Runmanager at the beginning of each 
    * event during the event loop.
    */
   virtual void GeneratePrimaries(G4Event * an_event);

private:
   G4ParticleGun * fparticle_gun;
};

#endif


