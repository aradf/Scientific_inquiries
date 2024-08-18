### Command line to instrument, compile and link
g++ -std=c++17 -I $G4INCLUDE -o G4FirstStep23 G4FirstStep23MainApp.cc -L $myname -Wl,-rpath $myname -lG4global -lG4ptl

g++ is the compiler.
-I is the flag for include files locations.
-o is the output file binary name.
-std is the c++ compiler set.
-L is the lib64 directory.


### useful environment variables.
echo $G4INSTALL
echo $G4INCLUDE

### Example for five day training.
file:///home/montecarlo/Desktop/geant4/g4_fiveday_training/geant4.lns.infn.it/pavia2023/introduction/index.html

### Book for the Application Developers
https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/index.html

### On line videos
https://videos.cern.ch/record/2299908
https://videos.cern.ch/search?page=1&size=21&q=G4FS24



