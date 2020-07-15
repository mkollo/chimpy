import importlib
import h5py
import re

from . import preprocess
importlib.reload(preprocess)
from . import ramdisk
importlib.reload(ramdisk)
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

class Raw:
    
    def __init__(self, file_path, ram_copy=True):
        self.original_file_path=file_path
        if ram_copy:
            self.file_path=ramdisk.create_clone(file_path)
        else:
            self.file_path=file_path
        self.fid=h5py.File(file_path, "r")
        self.traces=self.fid["sig"][:1024,:] 
        self.map=self.fid["mapping"]
        self.ids=[c[0] for c in self.map]
        self.electrodes=[c[1] for c in self.map]
        self.xs=[c[2] for c in self.map]
        self.ys=[c[3] for c in self.map]
        preprocess.filter_hdf_traces(self, 100, 9000, cmr=False)

    def filtered_filename(self):
        return re.sub(r"(?:\.raw\.h5){1,}$",".filt.h5",self.original_file_path)
    
    def __del__(self): 
        self.fid.close()
        ramdisk.wipe()
        
class Stim(Raw):
    
    def __init__(self, file_path):
        Raw.__init__(self,file_path)
        self.filt_traces = preprocess.filter_traces(self.traces, 100, 9000, cmr=False)
        self.amps = preprocess.get_spike_amps(self.filt_traces)
        self.collect_data()
        
    def collect_data(self):
        dtypes =  np.dtype([('channel', int),('electrode', int),('x', float),('y', float),('amp', float),('cluster', int)])
        self.data = np.empty(self.amps.shape[1], dtype=dtypes)
        self.data['channel']=self.amps[0]
        self.data['electrode']=np.take(self.electrodes,np.searchsorted(self.ids, self.data['channel']))
        self.data['x']=np.take(self.xs,np.searchsorted(self.ids, self.data['channel']))
        self.data['y']=np.take(self.ys,np.searchsorted(self.ids, self.data['channel']))
        self.data['amp']=self.amps[1]    
        self.data['cluster']=self.cluster_pixels()
      
    def cluster_pixels(self):   
        coords=np.transpose(np.vstack((self.data['x'],self.data['y'])))
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=35, linkage='ward').fit(coords)
        return clustering.labels_
    

            
class Noise(Raw):
    
    def __init__(self, file_path, stim_recording):
        Raw.__init__(self,file_path)
        self.data=stim_recording.data
        
    def filt_traces(self):
        self.filt_traces_batched = preprocess.filter_traces_batched(self.traces, 100, 7000, cmr=False)
        plt.plot(np.asarray(self.traces[4,800:1000],dtype=np.float32)-500)
        plt.plot(self.filt_traces_batched[4,800:1000])
        plt.show()
#         plt.magnitude_spectrum(self.traces[0,:], 20000)
#         plt.xlim([1,20000])
#         plt.ylim([0,0.25])
# #         print(self.traces[0,:])
#         plt.magnitude_spectrum(self.filt_traces[0,:],20000)
#         plt.show()

        