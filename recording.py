#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.                               

import importlib
import h5py
import re
import os

from . import preprocess
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class Recording:
    
    def __init__(self, filepath):
        self.filepath=filepath
        self.fid=h5py.File(filepath, "r")
        self.pixel_map=self.fid["mapping"]
        self.channels=np.array([c[0] for c in self.pixel_map])
        self.electrodes=np.array([c[1] for c in self.pixel_map])
        self.xs=np.array([c[2] for c in self.pixel_map])
        self.ys=np.array([c[3] for c in self.pixel_map])
        self.fid.close()
        self.filtered_filepath=re.sub(r"(?:\.raw\.h5){1,}$",".filt.h5",self.filepath)
        
    def remove_unconnected(self):
        connected=np.searchsorted(self.channels,self.connected_pixels)
        self.connected_in_mapping=connected
        self.channels=self.channels[connected]
        self.electrodes=self.electrodes[connected]
        self.xs=self.xs[connected]
        self.ys=self.ys[connected]
        
class StimRecording(Recording):
    
    def __init__(self, filepath, connected_threshold=50):
        Recording.__init__(self,filepath)
        fid=h5py.File(self.filepath, "r")
        self.filt_traces = preprocess.filter_traces(fid["sig"], 100, 9000, cmr=False, n_samples=20000)
        fid.close()
        self.amps = preprocess.get_spike_amps(self.filt_traces)        
        self.connected_pixels = np.where(self.amps>50)[0]
        self.unconnected_pixels = np.setdiff1d(np.array(range(1028)),self.connected_pixels)
        self.remove_unconnected()
        self.amps = self.amps[self.connected_pixels]
        self.clusters = self.cluster_pixels()        
        
    def cluster_pixels(self):   
        coords=np.transpose(np.vstack((self.xs,self.ys, self.amps)))
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=35, linkage='single').fit(coords)
        return clustering.labels_
    