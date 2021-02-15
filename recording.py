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

from numpy.lib.npyio import save
import h5py
import re
import os
from scipy.stats import normaltest
from . import preprocess
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
from tqdm import trange
import cusignal
import cupy as cp
class Recording:
    
    def __init__(self, filepath, *, savepath=None):
        self.filepath=filepath
        if savepath is None:
            self.savepath=filepath
        else:
            self.savepath = os.path.join(savepath, os.path.split(filepath)[1])
        self.filtered_filepath=re.sub(r"(?:\.raw\.h5){1,}$",".filt.h5",self.savepath)
        if os.path.isfile(self.filtered_filepath):
            self.filtered = True
            self.fid = h5py.File(self.filtered_filepath, 'r')
            self.first_frame = self.fid['first_frame'][0]
        else:
            self.filtered=False
            self.fid=h5py.File(filepath, "r")
            self.first_frame=self.fid["sig"][1027,0]<<16 | self.fid["sig"][1026,0]

        self.pixel_map=self.fid["mapping"]
        self.sample_length = self.fid['sig'].shape[1]
        if 'saturations' in self.fid.keys():
            self.saturations = np.sum(self.fid['saturations'], axis=1)
            #self.remove_saturation()
        if 'bits' in self.fid.keys():
            self.bits=self.fid["bits"]
        self.parse_mapping()
        self.fid.close()
        self.filtered_filepath=re.sub(r"(?:\.raw\.h5){1,}$",".filt.h5",self.savepath)
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
#        connected=self.channels[self.connected_pixels]
        # self.connected_in_mapping=connected
        self.channels, chan_index, connected_pixel_index = np.intersect1d(self.channels, self.connected_pixels, return_indices=True)
        #self.channels=self.channels[self.connected_pixels]
        self.connected_in_mapping = chan_index
        self.electrodes=self.electrodes[chan_index]
        self.xs=self.xs[chan_index]
        self.ys=self.ys[chan_index]

        
    def ttls(self):
        self.fid=h5py.File(self.filepath, "r")
        ttls = {}
        for i in range(len(self.fid["bits"])-1):
            ttl=self.fid["bits"][i]
            next_ttl=self.fid["bits"][i+1]
            start=ttl[0]
            stop=np.nan
            if next_ttl[1]==0:
                stop=next_ttl[0]
            if ttl[1] not in ttls:
                ttls[ttl[1]] = [[start, stop]]
            else:
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
        data=self.filtfid['sig'][self.channels,from_sample:to_sample]
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
    
    def calc_distance_matrix(self, eltimes=None):
        if self.distance_matrix==None:
            filtdata = self.filtered_data()
            if eltimes is None:
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
        
    def estim_distance_matrix(self, *, from_sample=0, to_sample=20000*60*5, dist_coef=143, dist_power=1/3):
        if self.estimated_coordinates is None:
            filtdata = self.filter_data(from_sample=from_sample, to_sample=to_sample)
            corrs = np.correlate(filtdata)
            corrs[corrs<0] == 1e-10
            distances=dist_coef/corrs**(dist_power) - dist_coef
            self.distance_matrix=distances
            return distances
    


    def calc_estimate_coordinates(self, dimensions=2):
        if self.estimated_coordinates==None:
            mds = manifold.MDS(n_components=dimensions, dissimilarity="precomputed", random_state=6)
            results = mds.fit(self.distance_matrix)
            estimated_coordinates=results.embedding_
            self.estimated_coordinates=estimated_coordinates
            return estimated_coordinates
    
    def remove_saturated(self, tol=0.01):
        sat_frac = self.saturations/self.sample_length < tol
        self.channels = self.channels[sat_frac]
        self.electrodes=self.electrodes[sat_frac]
        self.xs=self.xs[sat_frac]
        self.ys=self.ys[sat_frac]
        print('%d channels with saturation, removed %d channels with over %f saturation' % (len(np.where(self.saturations > 0)[0]), len(self.saturations) - len(self.channels), tol))       
    
    def remove_broken(self, from_sample=0, to_sample=65536):
        data = self.filtered_data(from_sample=from_sample, to_sample=to_sample)
        non_broken_chans = np.invert(np.isnan(normaltest(data, axis=1)[1]))
        self.channels = self.channels[non_broken_chans]
        self.xs = self.xs[non_broken_chans]
        self.electrodes = self.electrodes[non_broken_chans]
        self.ys = self.ys[non_broken_chans]
        print('removed %d channels showing abnormal distributions' % (len(data)-len(self.channels)))
        

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
    
    def find_tcs(self, chunk_size=65536, overlap=1000, order=40, threshold=4, return_stds=False):
        all_spikes = []
        spike_amps = []
        stds = np.std(self.filtered_data(from_sample=0, to_sample=chunk_size))
        for i in trange(round(self.sample_length/chunk_size)):
            chunk = self.filtered_data(from_sample=i*chunk_size, to_sample=(i+1)*chunk_size+overlap)
            chunk_spikes = cusignal.peak_finding.peak_finding.argrelmin(chunk, order=order, axis=1)
            spike_vals = chunk[chunk_spikes[0].get(), chunk_spikes[1].get()]
            sig_spikes = np.where(spike_vals <= - threshold*stds[chunk_spikes[0].get()])[0]
            all_spikes[0].append(chunk_spikes[0][sig_spikes])
            all_spikes[1].append(chunk_spikes[1][sig_spikes]+i*chunk_size)
            spike_amps.append(spike_vals[sig_spikes])
        all_spikes = cp.array([cp.concatenate(all_spikes[0]), cp.concatenate(all_spikes[1])])
        if return_stds:
            return all_spikes, spike_amps, stds
        else:
            return all_spikes, spike_amps


class StimRecording(Recording):
    
    def __init__(self, filepath, *, savepath=None, connected_threshold=10):
        Recording.__init__(self,filepath, savepath=savepath)
        fid=h5py.File(self.filepath, "r")
        self.filt_traces = preprocess.filter_traces(fid["sig"], 100, 9000, cmr=False, n_samples=20000)
        fid.close()
        self.amps = preprocess.get_spike_amps(self.filt_traces)        
        self.connected_pixels = np.where(self.amps>connected_threshold)[0]
        self.unconnected_pixels = np.setdiff1d(np.array(range(1028)),self.connected_pixels)
        self.remove_unconnected()
        self.amps = self.amps[self.connected_pixels]
        #self.clusters = self.cluster_pixels()        
        
    def cluster_pixels(self):   
        coords=np.transpose(np.vstack((self.xs,self.ys, self.amps)))
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=35, linkage='single').fit(coords)
        return clustering.labels_

class NoiseRecording(Recording):
    
    def __init__(self, filepath, stim_recording, *, savepath=None):
        Recording.__init__(self,filepath, savepath=savepath)
        fid=h5py.File(self.filepath, "r")
        if not os.path.isfile(self.filtered_filepath):
            preprocess.filter_experiment_local(self, stim_recording, 100, 9000, ram_copy = False, n_samples = 20000)
        fid=h5py.File(self.filtered_filepath, "r")
        self.noise_traces=fid['sig'][()]
        self.noises=np.std(self.noise_traces,axis=1)
        self.pixel_map=fid["mapping"]
        self.parse_mapping()
        fid.close()        
    