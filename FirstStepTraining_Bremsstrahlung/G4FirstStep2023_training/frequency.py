### Frequency Distribution Table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Run the Geant4 Simulator with the specified macro file.
### ./G4FirstStep23 ../g4macro.mac > output.log

# data_frame = pd.read_table("./output.log", 
#                            delimiter=',',
#                            names=['number', 'data'])

data_frame = pd.read_table("./histogram_energydeposit.dat", 
                           delimiter=',',
                           names=['number', 'data'])

x_data = data_frame.to_numpy().tolist()

d0_str = [item[0] for item in x_data]
d1_str = [item[1] for item in x_data]
d1_float = [float(item) for item in d1_str]

plt.plot(d1_float)
# plt.hist(d1_float, density=1, bins=50)
# plt.show()
plt.xlabel("Energy (KeV)")
plt.ylabel("y - label")
plt.title("Energy Deposit in 'W' due 100 KeV \nSimualted by Geant4")
plt.savefig('histogram.png')
plt.close()

print("hello world")







    