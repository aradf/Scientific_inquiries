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

#include "CRun.hh"
#include "CDetectorConstruction.hh"
#include "CPrimaryGeneratorAction.hh"

#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4EmCalculator.hh"
#include "G4Gamma.hh"

#include <iomanip>

CRun::CRun(CDetectorConstruction* detector_constructor) : G4Run(),
                                                          fdetector_constructor(detector_constructor), 
                                                          fparticle_definition(0), 
                                                          fkinetic_energy(0.0),
                                                          ftotal_count(0), 
                                                          fsum_track(0.0), 
                                                          fsum_trackSquared(0.0), 
                                                          fenergy_transfered(0.0)
{ 

}

CRun::~CRun()
{ 

}

void CRun::set_primary(G4ParticleDefinition* particle, G4double energy)
{ 
   fparticle_definition = particle;
   fkinetic_energy = energy;
}

void CRun::count_processes(G4String process_nameIdx) 
{
   /*
    * Standard Template Library has iterators which point to an element of std's container.  
    * in this case the std::map has pairs of G4String and G4int keys.  The find method of the 
    * std::map container returns an iterator object or pointer.  
    */
   // G4cout << "INFO: "   
   //        << process_nameIdx
   //        << G4endl;
   std::map<G4String,G4int>::iterator iterator_process = fproccess_counterMap.find(process_nameIdx);

   if ( iterator_process == fproccess_counterMap.end()) 
   {
      fproccess_counterMap[process_nameIdx] = 1;
   }
   else 
   {
      fproccess_counterMap[process_nameIdx]++; 
   }
}

void CRun::sum_tracks (G4double some_track)
{
   ftotal_count++;
   fsum_track += some_track;
   fsum_trackSquared += some_track * some_track;
}

void CRun::sum_energyTransfered (G4double energy)
{
   fenergy_transfered += energy;
}

void CRun::Merge(const G4Run* single_run)
{
    /*
     * 'static_casting' is explicite type conversion.  It converts a pointer of an object
     * to the a pointer of related class type.  Static casting is down/up.  The constant
     * single_run is a pointer of class type G4Run.  it is down casted (pointer conversion) to a 
     * local constant instance of class type CRun.
     */  
    const CRun* local_run = static_cast < const CRun * > ( single_run );

    // pass information about primary particle
    fparticle_definition = local_run->fparticle_definition;
    fkinetic_energy     = local_run->fkinetic_energy;
      
    //map: processes count
    std::map< G4String,G4int >::const_iterator iterator_process;
    for (iterator_process = local_run->fproccess_counterMap.begin(); 
         iterator_process != local_run->fproccess_counterMap.end(); 
         ++iterator_process) 
    {
       G4String process_nameIdx = iterator_process->first;
       G4int localCount  = iterator_process->second;
       if ( fproccess_counterMap.find( process_nameIdx ) == fproccess_counterMap.end()) 
       {
          fproccess_counterMap[process_nameIdx] = localCount;
       }
       else 
       {
          fproccess_counterMap[process_nameIdx] += localCount;
       }
    }
  
    ftotal_count += local_run->ftotal_count;
    fsum_track   += local_run->fsum_track;
    fsum_trackSquared  += local_run->fsum_trackSquared;
    fenergy_transfered += local_run->fenergy_transfered;
  
    G4Run::Merge( single_run ); 
} 

void CRun::end_ofSingleRun()
{
    G4int precision_value = 5;  
    G4int default_precision = G4cout.precision( precision_value );
  
    //run condition
    //        
    G4String particle_name    = fparticle_definition->GetParticleName();    
    G4Material* material = fdetector_constructor->get_material();
    G4double density     = material->GetDensity();
    G4double tickness    = fdetector_constructor->get_size();

    /*
     * numberOfEvent is a protected data member of G4Run class.
     */
    G4int local_numberOfEvent = this->GetNumberOfEvent();
    G4cout << "\n ======================== run summary ======================\n";
    G4cout << "\n The run is: " 
           << this->numberOfEvent 
           << " " 
           << particle_name 
           << " of "
           << G4BestUnit(fkinetic_energy,"Energy") 
           << " through " 
           << G4BestUnit(tickness,"Length") 
           << " of "
           << material->GetName() 
           << " (density: " 
           << G4BestUnit(density,"Volumic Mass") 
           << ")" 
           << G4endl;

    //frequency of processes
    G4int survive = 0;  
    G4cout << "\n Process calls frequency --->";
    std::map< G4String,G4int >::iterator iterator_process;  
    for ( iterator_process = fproccess_counterMap.begin(); 
          iterator_process != fproccess_counterMap.end(); 
          iterator_process++) 
    {
       G4String process_nameIdx = iterator_process->first;
       G4int count    = iterator_process->second;
       G4cout << "\t" << process_nameIdx << " = " << count;

       if ( process_nameIdx == "Transportation" ) 
           survive = count;
    }

    if (survive > 0) 
    {
       G4cout << "\n\n Nb of incident particles surviving after "
              << G4BestUnit(fdetector_constructor->get_size(),"Length") 
              << " of "
              << material->GetName() 
              << " : " 
              << survive 
              << G4endl;
    }

    if (ftotal_count == 0) 
       ftotal_count = 1;   //force printing anyway

    //compute mean free path and related quantities
    //
    G4double mean_freePath = fsum_track /ftotal_count;     
    G4double mean_trackSquared   = fsum_trackSquared/ftotal_count;     
    G4double rms = std::sqrt( std::fabs(mean_trackSquared - mean_freePath * mean_freePath));
    G4double cross_section = 1.0 / mean_freePath;     
    G4double massicMFP = mean_freePath * density;
    G4double massicCS  = 1.0 / massicMFP;

   G4cout << "\n\n mean_freePath:\t"   << G4BestUnit( mean_freePath, "Length" )
          << " +- "                    << G4BestUnit( rms,"Length")
          << "\tmassic: "              << G4BestUnit( massicMFP , "Mass/Surface")
          << "\n cross_section:\t"     << cross_section * cm << " cm^-1 "
          << "\t\t\tmassic: "          << G4BestUnit(massicCS, "Surface/Mass")
          << G4endl;
         
   //compute energy transfer coefficient
   //
   G4double mean_transfer   = fenergy_transfered / ftotal_count;
   G4double mass_transferCoefficient = massicCS * mean_transfer / fkinetic_energy;
   
   G4cout << "\n mean energy of charged secondaries: " 
          << G4BestUnit( mean_transfer , "Energy")
          << "\n     ---> mass_energy_transfer coef: "
          << G4BestUnit( mass_transferCoefficient , "Surface/Mass")
          << G4endl;       

   //check cross section from G4EmCalculator
   //
   G4cout << "\n Verification : "
          << "crossSections from G4EmCalculator \n";

   G4EmCalculator electromangentic_calculator;
   G4double sum_massSigma = 0.0;  
   for ( iterator_process = fproccess_counterMap.begin(); 
         iterator_process != fproccess_counterMap.end(); 
         iterator_process++) 
   {
       G4String process_nameIdx = iterator_process->first;      
       G4double mass_sigma = 
       electromangentic_calculator.GetCrossSectionPerVolume( fkinetic_energy, 
                                                             fparticle_definition,
                                                             process_nameIdx,
                                                             material ) / density;
     
       if ( fparticle_definition == G4Gamma::Gamma() )
            mass_sigma = electromangentic_calculator.ComputeCrossSectionPerVolume( fkinetic_energy,
                                                                                  fparticle_definition,
                                                                                  process_nameIdx,
                                                                                  material ) / density;
       sum_massSigma += mass_sigma;
       G4cout << "   " 
              << process_nameIdx 
              << "= " 
              << G4BestUnit( mass_sigma, 
                             "Surface/Mass");
    }
    G4cout << "   total= " 
           << G4BestUnit( sum_massSigma , "Surface/Mass") 
           << G4endl;

    // remove all contents in fproccess_counterMap 
    fproccess_counterMap.clear();

    //restore default format
    G4cout.precision( default_precision );  
}
