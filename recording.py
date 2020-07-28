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

import matplotlib.pyplot as plt

class Recording:
    
    def __init__(self, filepath):
        self.filepath=filepath
        self.fid=h5py.File(filepath, "r")
        self.map=self.fid["mapping"]
        self.channels=[c[0] for c in self.map]
        self.electrodes=[c[1] for c in self.map]
        self.xs=[c[2] for c in self.map]
        self.ys=[c[3] for c in self.map]
        self.fid.close()
        self.filtered_filepath=re.sub(r"(?:\.raw\.h5){1,}$",".filt.h5",self.filepath)
        
class StimRecording(Recording):
    
    def __init__(self, filepath):
        Recording.__init__(self,filepath)
        fid=h5py.File(self.filepath, "r")
        self.filt_traces = preprocess.filter_traces(fid["sig"], 100, 9000, cmr=False, n_samples=20000)
        fid.close()
        self.amps = preprocess.get_spike_amps(self.filt_traces[self.channels,:])
        self.clusters = self.cluster_pixels()
        self.strong_pixels, self.weak_pixels=self.strong_and_weak_pixels()
        
    def cluster_pixels(self):   
        coords=np.transpose(np.vstack((self.xs,self.ys)))
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=35, linkage='single').fit(coords)
        return clustering.labels_
    
    def strong_and_weak_pixels(self):
        strong_pixels=np.where(self.amps>50)[0]
        weak_pixels=np.where(self.amps<=50)[0]
        return strong_pixels, weak_pixels