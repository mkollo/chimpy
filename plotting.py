#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.
                                 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [8, 3]
import numpy as np

def plot_chip_surface_amps(stim_recording):  
    amps=np.array(stim_recording.amps)
    plt.scatter(stim_recording.xs,stim_recording.ys,c=amps, marker='s', s=1, cmap="magma_r")
    plt.xlim([0, 3900])
    plt.ylim([0, 2100])
    plt.clim([0, max(amps), ])
    plt.xlabel("distance (µm)")
    plt.ylabel("distance (µm)")
    plt.title("Chip surface - response amplitude to 1 mV spike")
    plt.colorbar()
    plt.show()

def plot_chip_surface_clusters(stim_recording):
    cmap = matplotlib.colors.ListedColormap(np.random.rand(156,3))
    plt.scatter(stim_recording.xs,stim_recording.ys,c=stim_recording.clusters, marker='s', s=1, cmap=cmap)
    plt.xlim([0, 3900])
    plt.ylim([0, 2100])
    plt.xlabel("distance (µm)")
    plt.ylabel("distance (µm)")
    plt.title("Chip surface - " + str(len(stim_recording.channels)) + " connected pixels, " + str(len(np.unique(stim_recording.clusters))) + " clusters")
    plt.colorbar()
    plt.show()
