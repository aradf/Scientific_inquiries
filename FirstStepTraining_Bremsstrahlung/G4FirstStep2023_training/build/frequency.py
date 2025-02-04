### Frequency Distribution Table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

### Run the Geant4 Simulator with the specified macro file.
### ./G4FirstStep23 ../g4macro.mac > output.log

def plot_histData():
    print("INFO: plot data ...")
    data_frame = pd.read_table("./histogram_energydeposit.dat", 
                            delimiter='\t',
                            names=['x', 'number', 'data'])

    x_data = data_frame.to_numpy().tolist()

    d0_str = [item[0] for item in x_data]
    d1_str = [item[1] for item in x_data]
    d2_str = [item[2] for item in x_data]
    d2_float = [float(item) for item in d2_str]

    plt.plot(d2_float)
    plt.xlabel("Energy (KeV)")
    plt.ylabel("Photon Count")
    plt.title("Primary particle Energy Deposit \n 'W' and 100 KeV Simualted by Geant4")
    plt.savefig('histogram.png')
    plt.close()

def plot_rawData():
    data_frame = pd.read_table("./output.log", 
                            delimiter=',',
                            names=['number', 'data'])

    x_data = data_frame.to_numpy().tolist()
    d0_str = [item[0] for item in x_data]
    d1_str = [item[1] for item in x_data]
    d1_float = [float(item) for item in d1_str]
    # d1_float = [float(item * 1000.0 + 20) for item in d1_float]

    # plt.hist(d1_float, density=1, bins=100)
    plt.hist(d1_float, bins=1000)
    plt.xlim([0,100])

    plt.xlabel("Energy (KeV)")
    plt.ylabel("Photon Count")
    plt.title("Primary particle Energy Deposit in \n 'W' and 100 KeV Simualted by Geant4")
    plt.savefig('histogram.png')
    plt.close()

# total arguments
argument_list = sys.argv[1:]
print(argument_list)

if "hist" in argument_list:
    plot_histData()

if "raw" in argument_list:
    plot_rawData()

print("hello world")







    