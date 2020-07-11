import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [8, 3]
import numpy as np

def plot_chip_surface_amps(stim_recording):
    data=stim_recording.data
    plt.scatter(data['x'],data['y'],c=data['amp'], marker='s', s=1, cmap="magma_r")
    plt.xlim([0, 3900])
    plt.ylim([0, 2100])
    plt.clim([0, max(data['amp']), ])
    plt.xlabel("distance (µm)")
    plt.ylabel("distance (µm)")
    plt.title("Chip surface - response amplitude to 1 mV spike")
    plt.colorbar()
    plt.show()

def plot_chip_surface_clusters(stim_recording):
    cmap = matplotlib.colors.ListedColormap(np.random.rand(156,3))
    data=stim_recording.data
    plt.scatter(data['x'],data['y'],c=data['cluster'], marker='s', s=1, cmap=cmap)
    plt.xlim([0, 3900])
    plt.ylim([0, 2100])
    plt.xlabel("distance (µm)")
    plt.ylabel("distance (µm)")
    plt.title("Chip surface - " + str(len(np.unique(data['cluster']))) + " clusters")
    plt.colorbar()
    plt.show()
