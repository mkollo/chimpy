import os
import subprocess
import time
from collections import deque
import numpy as np
from tqdm import tqdm
import hashlib

class Slurm:
    
    def __init__(self, job_name, n_tasks, gpu=False):
        self.job_name=job_name
        self.n_tasks=n_tasks
        self.gpu=gpu
        self.generate_shell_script()
        self.generate_out_file_names()
        self.generate_err_file_names()
        self.err_hashes=[]
    
    def generate_out_file_names(self):
        self.out_file_names=[]
        for i in range(self.n_tasks):
            self.out_file_names.append(self.job_name + "_" + str(i) + ".out")
        for of in self.out_file_names:
            if os.path.isfile(of):
                os.remove(of)
                subprocess.run(['touch', of])
                
    def generate_err_file_names(self):
        self.err_file_names=[]
        for i in range(self.n_tasks):
            self.err_file_names.append(self.job_name + "_" + str(i) + ".err")
        for ef in self.err_file_names:
            if os.path.isfile(ef):
                os.remove(ef)
                subprocess.run(['touch', ef])
        

    def clean_up_out_files(self):
        for of in self.out_file_names:
            if os.path.isfile(of):
                os.remove(of)
        for ef in self.err_file_names:
            if os.path.isfile(ef):
                os.remove(ef)

    def generate_shell_script(self):
        if os.path.isfile("slurm.sh"):
            os.remove("slurm.sh")
        file = open("slurm.sh", "w") 
        file.write("#!/bin/bash\n")
        file.write("#SBATCH --job-name=" + self.job_name + "\n")
        file.write("#SBATCH --ntasks=1\n")
        file.write("#SBATCH --nodes=1\n")
        file.write("#SBATCH --array=0-" + str(self.n_tasks-1) + "\n")
        file.write("#SBATCH --time=1:00:0\n")
        file.write("#SBATCH --mem=32G\n")
        if self.gpu:
            file.write("#SBATCH --partition=gpu\n")
            file.write("#SBATCH --gres=gpu:1\n")
        else:
            file.write("#SBATCH --partition=cpu\n")
        file.write("#SBATCH --output=" + self.job_name + "_%a.out\n")
        file.write("#SBATCH --error=" + self.job_name + "_%a.err\n")
        file.write("\n")
#         file.write("export OMPI_MCA_mpi_cuda_support=0\n")
        file.write("export OMPI_MCA_mpi_warn_on_fork=0\n")
        file.write("conda acticate chimpy-mpi &> /dev/null\n")
        file.write("module restore chimpy &> /dev/null\n")
        file.write("export OMPI_MCA_btl_openib_warn_nonexistent_if=0\n")
        file.write("srun --mpi=pmix_v3 /camp/home/kollom/working/mkollo/.conda/chimpy-mpi/bin/python chimpy/" + self.job_name + ".py ${SLURM_ARRAY_TASK_ID}")
        file.close()
    
    def get_progress(self, filename):
        try:
            lastline=subprocess.check_output(['tail', '-1', filename]).decode("utf-8")
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
        new_hashes=[(fname, hash_err(fname)) for fname in self.err_file_names]
        if not(self.err_hashes==new_hashes):
            for i, ef in enumerate(self.err_file_names):
                if os.path.isfile(ef):
                    with open (ef, "r") as errfile:
                        err_string=errfile.read()
                        if len(err_string)>0 and not(err_string.isspace()):
                            print('\33[91m'+err_string+'\033[0m')
        self.err_hashes=new_hashes
        
    def monitor(self):
        pbar = tqdm(total=1000*self.n_tasks, ncols=100)
        progress=0
        task_progresses = []
        for of in self.out_file_names:
            task_progresses.append(self.get_progress(of))
        while not(all(x==1000 for x in task_progresses)):
            time.sleep(0.1)
            for i, of in enumerate(self.out_file_names):
                task_progresses[i]=self.get_progress(of)
            if sum(task_progresses)>progress:
                pbar.update(sum(task_progresses)-progress)
                progress=sum(task_progresses)
                pbar.refresh()
            self.print_errors()
        print("Finished all Slurm jobs")   

    def run(self):
        process = subprocess.Popen(['sbatch', 'slurm.sh'])
        self.monitor()
        self.clean_up_out_files()

    def __del__(self):
        os.remove("slurm.sh")