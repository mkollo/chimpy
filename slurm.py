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
from subprocess import Popen, PIPE, check_output
import time
from collections import deque
import numpy as np
from tqdm import tqdm
import hashlib
import json
import time 

class Slurm:
    
    def __init__(self, job_name, nodes, gpu=False):
        self.job_name=job_name
        self.gpu=gpu
        self.nodes=nodes
        self.clean_up_out_files()
        self.generate_shell_script()
        self.err_hash=None
        if self.n_tasks()>0:
            self.kill_tasks()
        
    def clean_up_out_files(self):
        if os.path.isfile(self.job_name + ".out"):
            os.remove(self.job_name + ".out")
        if os.path.isfile(self.job_name + ".err"):
            os.remove(self.job_name + ".err")
        if os.path.isfile('params.json'):
            os.remove('params.json')
        pass

    def generate_shell_script(self):
        if os.path.isfile("slurm.sh"):
            os.remove("slurm.sh")
        file = open("slurm.sh", "w") 
        file.write("#!/bin/bash\n")
        file.write("#SBATCH --job-name=" + self.job_name + "\n")
        file.write("#SBATCH --ntasks="+str(self.nodes)+"\n")
        file.write("#SBATCH --nodes="+str(self.nodes)+"\n")
        file.write("#SBATCH --time=1:00:0\n")
        file.write("#SBATCH --mem=32G\n")
        if self.gpu:
            file.write("#SBATCH --partition=gpu\n")
            file.write("#SBATCH --gpus-per-node=1\n")
        else:
            file.write("#SBATCH --partition=cpu\n")
        file.write("#SBATCH --exclusive\n")
        file.write("#SBATCH --output=" + self.job_name + ".out\n")
        file.write("#SBATCH --error=" + self.job_name + ".err\n")
        file.write("\n")
        file.write("export OMPI_MCA_mpi_cuda_support=0\n")
        file.write("export OMPI_MCA_mpi_warn_on_fork=0\n")
        file.write("conda activate chimpy &> /dev/null\n")
        file.write("module restore chimpy &> /dev/null\n")
        file.write("export OMPI_MCA_btl_openib_warn_nonexistent_if=0\n")
        file.write("export OMPI_MCA_btl_openib_allow_ib=1\n")
        file.write("mpirun -np "+str(self.nodes)+" /camp/home/kollom/working/mkollo/.conda/chimpy/bin/python chimpy/" + self.job_name + ".py\n")
        file.close()

    def is_task_running(self):
        p = Popen(['squeue','--partition=gpu','--user=kollom','--noheader'], stdout=PIPE)
        output=p.communicate()[0].decode('utf-8').split()
        if len(output)>0:
            return output[4]=='R'
        else:
            return False
    
    def n_tasks(self):
        p = Popen(['squeue','--partition=gpu','--user=kollom','--noheader'], stdout=PIPE)
        return len(p.communicate()[0].decode('utf-8').split())
        
    def kill_tasks(self):
        p = Popen(['scancel','--partition=gpu','--user=kollom'], stdout=PIPE)
        output=p.communicate()
        n_gpu_tasks=self.n_tasks()
#         while n_gpu_tasks>0:
#             n_gpu_tasks=self.n_tasks()


    def get_progress(self):
        try:
            lastline=check_output(['tail', '-1', self.job_name + ".out"]).decode("utf-8")
            if lastline=="DONE\n":
                return 1000
            elif lastline=="":
                return 0
            else:
                return int(lastline)           
        except:
            return 0

    def print_errors(self):
        def hash_err(fname):
            if os.path.isfile(fname):
                with open(fname, 'rb') as file:
                    return hashlib.md5(file.read()).digest()               
            else:
                return ""
        new_hash=hash_err(self.job_name + ".err")
        if not(self.err_hash==new_hash):
            if os.path.isfile(self.job_name + ".err"):
                with open (self.job_name + ".err", "r") as errfile:
                    err_string=errfile.read()
                    if len(err_string)>0 and not(err_string.isspace()):
                        print('\33[91m'+err_string+'\033[0m')
                        print('didwriteerror')
        self.err_hash=new_hash
        
    def monitor(self, errors):
        pbar = tqdm(total=1000, ncols=100, position=0, leave=True)
        pbar.set_description("Setting up jobs")
        progress=0
        timeout=30
        start_time=time.time()
        tasks_running=False
        while not(progress==1000):           
            time.sleep(0.1)
            progress=self.get_progress()
            if progress>0:
                pbar.set_description("Filtering")
            pbar.update(progress)
            pbar.refresh()
            if errors:
                self.print_errors()
        print("Finished Slurm jobs")  

    def run(self, params, errors=True):
        self.kill_tasks()
        time.sleep(1)
        with open('params.json', 'w') as fp:
            json.dump(params, fp)
        process = Popen(['sbatch', 'slurm.sh'])
        if self.monitor(errors)==False:
            print("\33[91mERROR: Failed to start Slurm jobs\033[0m")
        self.clean_up_out_files()
        self.kill_tasks()
       
    def __del__(self):
        os.remove("slurm.sh")
        pass