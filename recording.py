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
importlib.reload(preprocess)
from . import ramdisk
importlib.reload(ramdisk)
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

class Recording:
    
    def __init__(self, file_path, ram_copy=True):
        self.ram_copy=ram_copy
        self.original_file_path=file_path
        if self.ram_copy:
            self.file_path=ramdisk.create_clone(file_path)
        else:
            self.file_path=file_path
        self.fid=h5py.File(file_path, "r")
        self.map=self.fid["mapping"]
        self.channels=[c[0] for c in self.map]
        self.electrodes=[c[1] for c in self.map]
        self.xs=[c[2] for c in self.map]
        self.ys=[c[3] for c in self.map]
        self.fid.close()
        self.filtered_filepath=re.sub(r"(?:\.raw\.h5){1,}$",".filt.h5",self.original_file_path)
    
    def filter_traces(self, stim_recording):
        if not(os.path.isfile(self.filtered_filepath)):
            self.filtfid=preprocess.filter_traces_parallel(self, stim_recording, 100, 9000, cmr=False)
        else:
            self.filtfid=h5py.File(self.filtered_filepath, "r")
            print("Filtered recording exists, loading from file...")

    def __del__(self): 
        ramdisk.wipe()
        
class Stim_recording(Recording):
    
    def __init__(self, file_path):
        Recording.__init__(self,file_path)
        fid=h5py.File(self.file_path, "r")
        self.filt_traces = preprocess.filter_traces(fid["sig"][:1024,:], 100, 9000, cmr=False)
        fid.close()
        self.amps = preprocess.get_spike_amps(self.filt_traces)
        self.collect_data()
        
    def collect_data(self):
        dtypes =  np.dtype([('channel', int),('electrode', int),('x', float),('y', float),('amp', float),('cluster', int)])
        self.data = np.empty(self.amps.shape[1], dtype=dtypes)
        self.data['channel']=self.amps[0]
        self.data['electrode']=np.take(self.electrodes,np.searchsorted(self.channels, self.data['channel']))
        self.data['x']=np.take(self.xs,np.searchsorted(self.channels, self.data['channel']))
        self.data['y']=np.take(self.ys,np.searchsorted(self.channels, self.data['channel']))
        self.data['amp']=self.amps[1]    
        self.data['cluster']=self.cluster_pixels()
      
    def cluster_pixels(self):   
        coords=np.transpose(np.vstack((self.data['x'],self.data['y'])))
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=35, linkage='ward').fit(coords)
        return clustering.labels_
    

