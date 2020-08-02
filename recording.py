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
        if 'bits' in self.fid.keys():
            self.bits=self.fid["bits"]
        self.parse_mapping()
        self.first_frame=self.fid["sig"][1027,0]<<16 | self.fid["sig"][1026,0]
        self.fid.close()
        self.filtered_filepath=re.sub(r"(?:\.raw\.h5){1,}$",".filt.h5",self.filepath)
    
    def filtered_data(self, from_sample, to_sample):
        self.fid=h5py.File(self.filtered_filepath, "r")
        data=self.fid['sig'][:,from_sample:to_sample][()]
        self.fid.close()
        return(data)
    
    def filter(self, stim_recording, low_cutoff=100, high_cutoff=9000, order=3, cmr=True, n_samples=-1, iron='local'):
        if iron=='local':
            preprocess.filter_experiment_local(self, stim_recording, low_cutoff, high_cutoff, order=3, cmr=False, n_samples=-1) 

    def parse_mapping(self):
        self.channels=np.array([c[0] for c in self.pixel_map])
        self.electrodes=np.array([c[1] for c in self.pixel_map])
        self.xs=np.array([c[2] for c in self.pixel_map])
        self.ys=np.array([c[3] for c in self.pixel_map])  
        
    def remove_unconnected(self):
        connected=np.searchsorted(self.channels,self.connected_pixels)
        self.connected_in_mapping=connected
        self.channels=self.channels[connected]
        self.electrodes=self.electrodes[connected]
        self.xs=self.xs[connected]
        self.ys=self.ys[connected]
        
    def ttls(self):
        self.fid=h5py.File(self.filepath, "r")
        ttls = {k: [] for k in range(33)}
        for i in range(len(self.fid["bits"])-1):
            ttl=self.fid["bits"][i]
            next_ttl=self.fid["bits"][i+1]
            start=ttl[0]
            stop=np.nan
            if next_ttl[1]==0:
                stop=next_ttl[0]
            ttls[ttl[1]].append([start, stop])
        ttls.pop(0, None)
        cleaned_ttls={}
        for k, t in ttls.items():
            if len(t)>0:
                t=(np.array(t)-self.first_frame)/20000
                t=np.append(t,np.round((t[:,1]-t[:,0])[None,].T*10-1),1)
                cleaned_ttls[k]=t
        self.fid.close()
        return cleaned_ttls
        
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

class NoiseRecording(Recording):
    
    def __init__(self, filepath, stim_recording):
        Recording.__init__(self,filepath)
        fid=h5py.File(self.filepath, "r")
        preprocess.filter_experiment_local(self, stim_recording, 100, 9000, ram_copy = True, n_samples = 20000)
        fid=h5py.File(self.filtered_filepath, "r")
        self.noise_traces=fid['sig'][()]
        self.noises=np.std(self.noise_traces,axis=1)
        self.pixel_map=fid["mapping"]
        self.parse_mapping()
        fid.close()        
    