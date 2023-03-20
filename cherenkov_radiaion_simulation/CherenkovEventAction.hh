#ifndef CHERENKOV_EVENT_HH
#define CHERENKOV_EVENT_HH

#include "G4UserEventAction.hh"
#include "G4Event.hh"
#include "G4AnalysisManager.hh"
#include "CherenkovRunAction.hh"

/**
 * Class description.
 * This class inherits from the G4UserEventAction such that the 
 * total energy deposited can be passed to an analys manager object.
 *
 * The two methods BeginOfEventAction() and EndOfEventAction() are invoked
 * at the beginning and the end of one event processing by G4EventManager.
 * Note: the BeginOfEventAction() is invoked when a G4Event object is
 * sent to G4EventManager, consequently the primary vertexes/particles have already
 * been constructed by the primary generator. In case, something must be done 
 * prior to generating primaries, it must be done in the G4VUserPrimaryGeneratorAction class.
 */
class CCherenkovEventAction : public G4UserEventAction
{

public:
   CCherenkovEventAction(CCherenkovRunAction *);
   ~CCherenkovEventAction();

   /**
    * This method is invoked by G4EventManager for each event.
    * It is invoked before converting the primary partiles to G4Track object.
    */
   virtual void BeginOfEventAction(const G4Event *);

   /**
    * This method is invoked by G4EventManager for each event.
    * it is invoked at the end of event processing.  
    */
   virtual void EndOfEventAction(const G4Event *);

   void add_energy_deposite(G4double energy_deposite) {fEdep += energy_deposite;}

private:
   G4double fEdep;
};

#endif