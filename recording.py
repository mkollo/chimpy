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
from sklearn import manifold

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
        self.distance_matrix=None
        self.estimated_coordinates=None
        
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
    
    def filtered_data(self, from_sample=-1, to_sample=-1):
        self.filtfid=h5py.File(self.filtered_filepath, "r")
        if from_sample==-1:
            from_sample=0
        if to_sample==-1:
            to_sample=self.filtfid['sig'].shape[1]
        data=self.filtfid['sig'][:,from_sample:to_sample][()]
        self.filtfid.close()
        return(data)
    
    def elstim_times(self, from_sample=-1, to_sample=-1):
        self.filtfid=h5py.File(self.filtered_filepath, "r")
        minsig=np.min(self.filtfid['sig'],axis=0)[None,:]
        if from_sample==-1:
            from_sample=0
        if to_sample==-1:
            to_sample=self.filtfid['sig'].shape[1]
        elstim_times=preprocess.get_spike_crossings(minsig, 1)
        elstim_times=elstim_times[np.insert(np.isclose(np.diff(elstim_times),10000, atol=500),0,False)]        
        self.filtfid.close()
        elstim_times=elstim_times[elstim_times>from_sample]
        elstim_times=elstim_times[elstim_times<to_sample]
        elstim_times=elstim_times
        group_indices=np.where(np.diff(elstim_times)>12000)[0]+1
        return np.split(elstim_times, group_indices)
    
    def calc_distance_matrix(self):
        if self.distance_matrix==None:
            filtdata=self.filtered_data()
            eltimes=self.elstim_times()
            stim_response_map=np.array([np.mean(filtdata[:,eltimes[x]],axis=1) for x in range(len(eltimes))])
            self.stim_response_map=stim_response_map
            stim_pixels=np.argmax(np.abs(stim_response_map[:,:]),axis=1)
            distance_matrix=np.full((stim_response_map.shape[1],stim_response_map.shape[1]),np.nan)
            for stim_type in np.unique(stim_pixels):
                stim_trials=np.where(stim_pixels==stim_type)
                distance_matrix[stim_type,:]=np.mean(1/np.abs(stim_response_map[stim_trials,:]),axis=1)[0,:]*5000
            distance_matrix[distance_matrix>5000]=5000
            distance_matrix[distance_matrix<-5000]=np.nan
            filled_rows=np.where(np.all(np.isnan(distance_matrix),axis=1))[0]
            distance_matrix[filled_rows,:]=distance_matrix[:,filled_rows].T
            mean_distance=np.nanmedian(distance_matrix)
            distance_matrix[np.isnan(distance_matrix)]=mean_distance
            distance_matrix=(distance_matrix + distance_matrix.T)/2        
            self.distance_matrix=distance_matrix
            return distance_matrix

    def calc_estimate_coordinates(self, dimensions=2):
        if self.estimated_coordinates==None:
            mds = manifold.MDS(n_components=dimensions, dissimilarity="precomputed", random_state=6)
            results = mds.fit(self.distance_matrix)
            estimated_coordinates=results.embedding_
            self.estimated_coordinates=estimated_coordinates
            return estimated_coordinates
        

    def write_probe_file(filename, coords, radius):
        with open(filename, "w") as fid:
            fid.write("total_nb_channels = "+str(coords.shape[0])+"\n")
            fid.write("radius = "+str(radius)+"\n")        
            fid.write("channel_groups = {\n")
            fid.write("\t1: {\n")
            fid.write("\t\t'channels': list(range("+srt(coords.shape[0])+")),\n")
            fid.write("\t\t'graph': [],\n")
            fid.write("\t\t'geometry': {\n")
            for i in range(coords.shape[0]):
                fid.write("\t\t\t"+str(i)+":  [  "+', '.join([str(c) for c in coords[i,:]])+"],\n")
            fid.write("\t\t}\n")
            fid.write("\t}\n")
            fid.write("}\n")


class StimRecording(Recording):
    
    def __init__(self, filepath, connected_threshold=50):
        Recording.__init__(self,filepath)
        fid=h5py.File(self.filepath, "r")
        self.filt_traces = preprocess.filter_traces(fid["sig"], 100, 9000, cmr=False, n_samples=20000)
        fid.close()
        self.amps = preprocess.get_spike_amps(self.filt_traces)        
        self.connected_pixels = np.where(self.amps>25)[0]
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
    