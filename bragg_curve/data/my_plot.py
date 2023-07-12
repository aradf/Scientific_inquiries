"""
Python script plots the individual Bragg peaks corresponding to Porton beam from 25 to 75 [MeV].
dose distributions (energy deposites) for therapeutic proton beams
"""

import matplotlib.pyplot as plt
import numpy as np
 

def plot_all():
    print("Plot all!")
    X25, Y25 = np.loadtxt('task5_25.csv', delimiter=',', unpack=True)
    plt.plot(X25, Y25, 'r', label='25 MeV')

    X35, Y35 = np.loadtxt('task5_35.csv', delimiter=',', unpack=True)
    plt.plot(X35, Y35, 'b', label='35 MeV')
    
    X45, Y45 = np.loadtxt('task5_45.csv', delimiter=',', unpack=True)
    plt.plot(X45, Y45, 'g', label='45 MeV')

    X55, Y55 = np.loadtxt('task5_55.csv', delimiter=',', unpack=True)
    plt.plot(X55, Y55, 'm', label='55 MeV')

    X65, Y65 = np.loadtxt('task5_65.csv', delimiter=',', unpack=True)
    plt.plot(X65, Y65, 'y', label='65 MeV')

    X75, Y75 = np.loadtxt('task5_75.csv', delimiter=',', unpack=True)
    plt.plot(X75, Y75, 'c', label='75 MeV')


    plt.title('Bragg Peak for 25-75 MeV Proton - using Geant4')
    plt.xlabel('Range in [cm]')
    plt.ylabel('Energy Deposition in [keV]')
    plt.legend()
    plt.savefig('task5_all.png')
    plt.show()


if __name__ == "__main__":
    plot_all()

