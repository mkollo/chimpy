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
        self.select_rec(brain_selection,'brain')
        self.print_all_rec_lists()
        self.recordings['stim']=Stim(self.paths['stim'][self.selections['stim']])
        self.recordings['noise']=Noise(self.paths['stim'][self.selections['stim']], self.recordings['stim'])
        plot_chip_surface_amps(self.recordings['stim'])
        plot_chip_surface_clusters(self.recordings['stim'])
        self.recordings['noise'].filt_traces()
        
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
        
    def print_all_rec_lists(self):
        print("Stim  recordings:")
        self.print_rec_list('stim')
        print("Noise recordings:")
        self.print_rec_list('noise')
        print("Brain recordings:")
        self.print_rec_list('brain')

        
    def print_file_item(self, selected, n, file_path):
        print('[{0}] {1:3}: {2}'.format(' ' if selected==False else 'x', str(n), os.path.basename(file_path)))