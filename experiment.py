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
import numpy as np
from chimpy.recording import Recording, StimRecording, NoiseRecording
from PIL import Image

class Experiment:
    '''
    A series of chimpy recordings associated with a single experimental day
    '''

    
    def __init__(self, experiment_path, *, savepath=None, stim_selection=-1, conn_selection=-1, noise_selection=-1, pid_selection=-1, smr_selection=-1, no_graphics=False, switched_recordings=[None], ignore_recordings=[None]):
        '''
        Arguments:
        experiment_path - The path to the head folder containing the folders for brain, conn etc.

        Optional arguments:
        savepath - If the location for new additional files created is different to the path that the original experimental files are containted, default None
        stim_selection - Index of the stim recordings to select for use, default -1
        conn_selection - Index of the conn recordings to select for use, default -1
        noise_selection - Index of the noise recordings to select for use, default -1
        pid_selection - Index of the pid recording (if present) to select for use, default -1
        smr_selection - Index of the spike2 recording to use
        no_graphics - If a graphical output should be shown outlying noise/connected pixels, default False
        switched_recordings - If any recordings need to be moved between groups, is passed sets of 3 values: intial group, target group, and inital index
        ignore_recordings - If any recording groups should be ignored, use at own risk.
        '''
        self.experiment_path=experiment_path
        self.savepath=savepath
        self.paths={}
        self.selections={}
        self.recordings={}
        self.explore_paths(self.experiment_path)
        self.switch_recording_group(switched_recordings)

        self.select_rec('stim', stim_selection)
        self.select_rec('noise', noise_selection)
        self.select_rec('pid', pid_selection)
        self.select_rec('smr', smr_selection)
        self.select_rec('conn', conn_selection)
        self.unselect_rec('brain')
        print('Conn recordings')
        self.print_rec_list('conn')
        print("Stim  recordings:")
        self.print_rec_list('stim')
        print("Noise recordings:")
        self.print_rec_list('noise')        
        print("PID recordings:")
        self.print_rec_list('pid')
        print("Spike2 recordings:")
        self.print_rec_list('smr')
        # self.ct_shape=np.array(Image.open(self.paths['ct'][0])).shape + tuple([len(self.paths['ct'])])
        # print("CT found ===== shape(W,L,H) = " + str(self.ct_shape))
        print("Brain recordings:")
        self.print_rec_list('brain')
        # print("Loading stim recording...")
        # self.recordings['stim']=StimRecording(self.paths['stim'][self.selections['stim']], savepath=savepath)
        print('loading conn recording...')
        if 'conn' not in ignore_recordings:
            self.recordings['conn']=StimRecording(self.paths['conn'][self.selections['conn']], savepath=savepath)
            self.connected_pixels=self.recordings['conn'].connected_pixels
            self.unconnected_pixels=self.recordings['conn'].unconnected_pixels
        else:
            print('Ignored conn recording, WARNING: This will almost inevitably cause crashes/bugs')
        print("Loading noise recording...")
        if 'noise' not in ignore_recordings:
            self.recordings['noise']=NoiseRecording(self.paths['noise'][self.selections['noise']], self.recordings['conn'], savepath=savepath)
        else:
            print('Ignored noise recording')        
        print("Calculating pixel amps...")
        if not no_graphics:
            from chimpy.plotting import plot_chip_surface_amps, plot_chip_surface_clusters, plot_noise_histogram, plot_chip_surface_noises, create_figure
            fig, axs = create_figure(2,2)
            plot_chip_surface_amps(self.recordings['conn'], fig, axs[0][0])
            plot_chip_surface_clusters(self.recordings['conn'], fig, axs[0][1])
            plot_noise_histogram(self.recordings['noise'], fig, axs[1][0])
            plot_chip_surface_noises(self.recordings['noise'], fig, axs[1][1])


    def select_brain_recording(self, selection):
        self.select_rec('brain', selection)
        self.recordings['brain']=Recording(self.paths['brain'][self.selections['brain']], savepath=self.savepath)
        self.recordings['brain'].connected_pixels = self.recordings['conn'].connected_pixels
        self.recordings['brain'].unconnected_pixels = self.recordings['conn'].unconnected_pixels
        #self.recordings['brain'].parse_mapping()
        self.print_rec_list('brain')
        if self.recordings['brain'].filtered:
            self.recordings['brain'].remove_broken()
            self.recordings['brain'].remove_saturated()
        
    def unselect_rec(self, rec_type):
        self.selections[rec_type]=-1
        
    def select_rec(self, rec_type='brain', selection=-1):
        if selection<0:
            self.selections[rec_type]=len(self.paths[rec_type])-1
        else:
            self.selections[rec_type]=selection
            
    def explore_paths(self, experiment_folder):
        self.paths['conn']=glob.glob(self.experiment_path+"/conn/*.raw.h5")
        self.paths['conn'].sort(key=os.path.getmtime)
        self.paths['stim']=glob.glob(self.experiment_path+"/stim/*.raw.h5")
        self.paths['stim'].sort(key=os.path.getmtime)
        self.paths['noise']=glob.glob(self.experiment_path+"/noise/*.raw.h5")
        self.paths['noise'].sort(key=os.path.getmtime)
        self.paths['brain']=glob.glob(self.experiment_path+"/brain/*.raw.h5")
        self.paths['brain'].sort(key=os.path.getmtime)
        self.paths['smr']=glob.glob(self.experiment_path+"/*[!_PID].smr*")
        self.paths['smr'].sort(key=os.path.getmtime)
        self.paths['ct']=glob.glob(self.experiment_path+"/ct/*.tiff")
        self.paths['ct'].sort()
        self.paths['pid']=glob.glob(self.experiment_path+"/*_PID.smr")
        self.paths['pid'].sort(key=os.path.getmtime)
        
    def print_rec_list(self, rec_type):
        for i in range(len(self.paths[rec_type])):
            self.print_file_item(i==self.selections[rec_type],i,self.paths[rec_type][i])
        print("")
        
    def print_file_item(self, selected, n, filepath):
        file_size=os.stat(filepath).st_size/(1024*1024*1024.0)
        print('[{0}] {2:3} {1:5.1f}GB ––– {3}'.format(' ' if selected==False else 'x', file_size, str(n), os.path.basename(filepath)))

    def switch_recording_group(self, switches):
        for switch in switches:
            if switch is None:
                print('No recording switches')
                break
            else:
                initial_group, new_group, position = switch
                assert initial_group in self.paths and new_group in self.paths
                exp_path = self.paths[initial_group][position]
                del self.paths[initial_group][position]
                self.paths[new_group].append(exp_path)
                print('Switched recording %s from %s to %s' % (os.path.basename(exp_path), initial_group, new_group))
