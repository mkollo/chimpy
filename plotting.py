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
import numpy as np
from chimpy.preprocess import power_spectrum

import matplotlib.pyplot as plt
from sciplotlib import style as spstyle

def create_figure(x, y, nature_style=False):
    matplotlib.rcParams['figure.figsize'] = [7*x, 4*y]
    if nature_style:
        with plt.style.context(spstyle.get_style('nature')):
            fig, axs = plt.subplots(y,x, figsize=(6*x,3*y))
    else:
            fig, axs = plt.subplots(y,x, figsize=(6*x,3*y))
    return fig, axs
    
def plot_chip_surface_amps(stim_recording, fig, ax):
    amps=np.array(stim_recording.amps)
    scatter=ax.scatter(stim_recording.xs,stim_recording.ys,c=amps, marker='s', s=1, cmap="magma_r")
    ax.set_xlim([0, 3900])
    ax.set_ylim([0, 2100])
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("distance (µm)")
    ax.set_title("Chip surface - response amplitude to 1 mV spike")
    cbar=fig.colorbar(scatter, ax=ax)
    scatter.set_clim([0, max(amps), ])


def plot_chip_surface_clusters(stim_recording, fig, ax):
    cmap = matplotlib.colors.ListedColormap(np.random.rand(156,3))
    scatter=ax.scatter(stim_recording.xs,stim_recording.ys,c=stim_recording.clusters, marker='s', s=1, cmap=cmap)
    ax.set_xlim([0, 3900])
    ax.set_ylim([0, 2100])
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("distance (µm)")
    ax.set_title("Chip surface - " + str(len(stim_recording.channels)) + " connected pixels, " + str(len(np.unique(stim_recording.clusters))) + " clusters")
    cbar=fig.colorbar(scatter, ax=ax)

def plot_chip_surface_noises(noise_recording, fig, ax):
    noises=np.array(noise_recording.noises)
    scatter=ax.scatter(noise_recording.xs,noise_recording.ys,c=noises, marker='s', s=1, cmap="magma_r")
    ax.set_xlim([0, 3900])
    ax.set_ylim([0, 2100])
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("distance (µm)")
    ax.set_title("Chip surface - noise levels (µV)")
    cbar=fig.colorbar(scatter, ax=ax)
    scatter.set_clim([0, max(noises), ])

def plot_noise_histogram(noise_recording, fig, ax):    
    hist=ax.hist(noise_recording.noises, 25, edgecolor='black', linewidth=1)
    ax.set_xlabel("RMS noise (µV)")
    ax.set_ylabel("pixels")
    plt.tight_layout(pad=1)
    
def plot_power_spectrum(data1, data2=None):
    spectrum1=power_spectrum(data1)
    plt.plot(spectrum1[0][10:], spectrum1[1][10:])
    if data2 is not None:
        spectrum2=power_spectrum(data2)
        plt.plot(spectrum2[0][10:], spectrum2[1][10:])
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    

def plot_amplitude_vs_noise(stim_recording, noise_recording, fig, ax):    
        ax.scatter(stim_recording.amps,noise_recording.noises)
        ax.set_xlabel("Amplitudes (pixel value)")
        ax.set_ylabel("RMS noise (µV)")