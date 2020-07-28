#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.
                                 
import glob
import os
from chimpy import *

class Experiment:
    
    def __init__(self, experiment_folder, stim_selection=-1, noise_selection=-1, brain_selection=-1):
        self.base_dir = "/camp/home/kollom/working/mkollo/CHIME/"
        self.paths={}
        self.selections={}
        self.recordings={}
        self.explore_paths(experiment_folder)
        self.select_rec(stim_selection,'stim')
        self.select_rec(noise_selection,'noise')
        self.unselect_rec('brain')
        print("Stim  recordings:")
        self.print_rec_list('stim')
        print("Noise recordings:")
        self.print_rec_list('noise')
        print("Loading stim recording...")
        self.recordings['stim']=Stim_recording(self.paths['stim'][self.selections['stim']])
        self.strong_pixels=self.recordings['stim'].amps[0].astype(int)
        self.weak_pixels=sorted(list(set(self.recordings['stim'].channels)-set(self.strong_pixels)))
        print("Loading noise recording...")
        self.recordings['noise']=Recording(self.paths['noise'][self.selections['noise']])
        print("Calculating pixel amps...")
        plot_chip_surface_amps(self.recordings['stim'])
        plot_chip_surface_clusters(self.recordings['stim'])
        print("Brain recordings:")
        self.print_rec_list('brain')
        
    def load_brain_recording(self, selection, ram_copy=True):
        self.recordings['brain']=Recording(self.paths['brain'][selection], ram_copy)
        
    def unselect_rec(self, rec_type):
        self.selections[rec_type]=-1
        
    def select_rec(self, selection, rec_type):
        if selection<0:
            self.selections[rec_type]=len(self.paths[rec_type])-1
        else:
            self.selections[rec_type]=selection
            
    def explore_paths(self, experiment_folder):
        self.paths['stim']=glob.glob(self.base_dir+experiment_folder+"/stim/*.raw.h5")
        self.paths['stim'].sort(key=os.path.getmtime)
        self.paths['noise']=glob.glob(self.base_dir+experiment_folder+"/noise/*.raw.h5")
        self.paths['noise'].sort(key=os.path.getmtime)
        self.paths['brain']=glob.glob(self.base_dir+experiment_folder+"/brain/*.raw.h5")
        self.paths['brain'].sort(key=os.path.getmtime)
        
    def print_rec_list(self, rec_type):
        for i in range(len(self.paths[rec_type])):
            self.print_file_item(i==self.selections[rec_type],i,self.paths[rec_type][i])
        print("")
        
    def print_file_item(self, selected, n, file_path):
        file_size=os.stat(file_path).st_size/(1024*1024*1024.0)
        print('[{0}] {2:3} {1:5.1f}GB ––– {3}'.format(' ' if selected==False else 'x', file_size, str(n), os.path.basename(file_path)))