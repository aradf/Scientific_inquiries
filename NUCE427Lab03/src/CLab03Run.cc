#include "G4SystemOfUnits.hh"

#include "CLab03Run.hh"
#include "CLab03DetectorConstruction.hh"
#include "CLab03PrimaryGeneratorAction.hh"

namespace NUCE427LAB03
{

CLab03Run::CLab03Run(CLab03DetectorConstructor*   detector_constructor,
                     CLab03PrimaryGeneratorAction* primary_generator) : G4Run()
{

   fdetector_constructor = detector_constructor;
   fprimary_generator = primary_generator;

   // energy deposit in the target min == 0.0, max == 200.0; bin numbers == 300;
   fenergy_depositHistogram = new Hist("histogram_energydeposit.dat", 
                                       0.0 * CLHEP::keV, 
                                       200.0 * CLHEP::keV, 
                                       10);
}

CLab03Run::~CLab03Run()
{
   if ( fenergy_depositHistogram != nullptr)
   {
       delete fenergy_depositHistogram;
       fenergy_depositHistogram = nullptr;
   }
}

void CLab03Run::Merge(const G4Run * current_run)
{
   // static cast performs data conversion between the G4Run parent class to CLab03Run
   // child class.  The two classes are related, the static cast is performing down casting.
   // It does NOT perform a check to make sure the data conversion is sucessful.
   const CLab03Run * other_run = static_cast<const CLab03Run *> (current_run);

   if (other_run != nullptr)
       fenergy_depositHistogram->Add( other_run->fenergy_depositHistogram );
   else
       G4cout << "   --> the static cast failed ..." << G4endl;

   this->fprimary_generator = other_run->fprimary_generator;

   /*
    * The G4Run's merge has to be invoked as well ...
    * CLab03Run's merge has occured, before this line of instruction.  
    */
   G4Run::Merge( current_run );
}

void CLab03Run::EndOfRunSummary()
{

   const G4int number_events = GetNumberOfEvent();
   if ( number_events == 0) 
      return;

   // G4cout << "==== End of Run ==== \n"
   //        << "     Number of Events = "
   //        << number_events
   //        << "\n"
   //        << "     Target Thickness = "
   //        << fdetector_constructor->get_targetThickness() / CLHEP::mm 
   //        << " (mm)"
   //        << "\n"
   //        << G4endl;
   
   fenergy_depositHistogram->WriteToFile( true );
}

};
