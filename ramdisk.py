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

class RamFile:
    
    def __init__(self, filepath, tmpfs_path="/dev/shm", mode="r"):
        self.mode=mode
        self.tmpfs_path=tmpfs_path
        self.filepath=filepath
        if not(os.path.isdir(self.tmpfs_path+"/chimpy_temp")):
            os.mkdir(self.tmpfs_path+"/chimpy_temp")
        self.ram_filepath=self.tmpfs_path+"/chimpy_temp/"+os.path.basename(filepath)
        if mode=="r":
            shutil.copy(filepath, self.ram_filepath)
            os.chmod(self.ram_filepath, 0o755)

    def save():
        shutil.copy(self.ram_filepath, self.filepath)

    def __del__():
        shutil.rmtree(self.tmpfs_path+"/chimpy_temp", ignore_errors=True)