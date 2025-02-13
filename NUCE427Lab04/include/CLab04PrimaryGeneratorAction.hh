#ifndef PRIMARY_GENERATOR_HH
#define PRIMARY_GENERATOR_HH

#include <G4VUserPrimaryGeneratorAction.hh>

class G4Event;
class G4ParticleGun;

namespace NUCE427LAB04
{

class CLab04PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    CLab04PrimaryGeneratorAction();
    ~CLab04PrimaryGeneratorAction();
    virtual void GeneratePrimaries(G4Event * event);    

private:
    G4ParticleGun* fparticle_gun;

};


}

#endif