#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.
                                 
import os
import shutil
import h5py

def create_clone(filepath):
    if not(os.path.isdir("/dev/shm/chime_temp")):
        os.mkdir("/dev/shm/chime_temp")
    memfilepath="/dev/shm/chime_temp/"+os.path.basename(filepath)
    shutil.copy(filepath, memfilepath)
    os.chmod(memfilepath, 0o755)
    return memfilepath

def create_hdf_ramfile(filepath):
    if not(os.path.isdir("/dev/shm/chime_temp")):
        os.mkdir("/dev/shm/chime_temp")
    memfilepath="/dev/shm/chime_temp/"+os.path.basename(filepath)
    fid=h5py.File(memfilepath, 'w')
    os.chmod(memfilepath, 0o755)
    return fid

def save_ramfile(filepath):
    memfilepath="/dev/shm/chime_temp/"+os.path.basename(filepath)
    shutil.copy(memfilepath, filepath)
    os.chmod(memfilepath, 0o755)

def wipe():
    shutil.rmtree("/dev/shm/chime_temp", ignore_errors=True)
