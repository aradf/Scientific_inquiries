#include "G4SystemOfUnits.hh"

#include "CFirstStepRun.hh"
#include "CFirstStepDetectorConstruction.hh"
#include "CFirstStepPrimaryGeneratorAction.hh"

namespace FS
{

CFirstStepRun::CFirstStepRun(CFirstStepDetectorConstruction*   detector_construction,
                             CFirstStepPrimaryGeneratorAction* primary_generator) 
                             : G4Run()
{

   fdetector_construction = detector_construction;
   fprimary_generator = primary_generator;

   // energy deposit in the target min == 0.0, max == 10.0; bin numbers == 50;
   fenergy_depositHistogram = new Hist("histogram_energydeposit.dat", 
                                       0.0, 
                                       20.0 * CLHEP::keV, 
                                       100);
}

CFirstStepRun::~CFirstStepRun()
{
   if ( fenergy_depositHistogram != nullptr)
   {
       delete fenergy_depositHistogram;
       fenergy_depositHistogram = nullptr;
   }
}

void CFirstStepRun::Merge(const G4Run * current_run)
{
   // static cast performs data conversion between the G4Run parent class to CFirstStepRun
   // child class.  The two classes are related, the static cast is performing down casting.
   // It does NOT perform a check to make sure the data conversion is sucessful.
   const CFirstStepRun * other_run = static_cast<const CFirstStepRun *> (current_run);

   if (other_run != nullptr)
      fenergy_depositHistogram->Add( other_run->fenergy_depositHistogram );
   else
      G4cout << "   --> the static cast failed ..."
             << G4endl;

   this->fprimary_generator = other_run->fprimary_generator;

   /*
    * The G4Run's merge has to be invoked as well ...
    * CFirstStepRun's merge has occured, before this line of instruction.  
    */
   G4Run::Merge( current_run );
}

void CFirstStepRun::EndOfRunSummary()
{
   const G4int number_events = GetNumberOfEvent();
   if ( number_events == 0)
      return;

   G4cout << "==== End of Run ==== \n"
          << "     Number of Events = "
          << number_events
          << "\n"
          << "     Target Thickness = "
          << fdetector_construction->get_targetThickness() / CLHEP::mm 
          << " (mm)"
          << "\n"
          << G4endl;
   
   fenergy_depositHistogram->WriteToFile( true );
}

};

