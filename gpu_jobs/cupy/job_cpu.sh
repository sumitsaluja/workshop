#!/bin/bash -l
#SBATCH --job-name=cupy-gpu      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)

module load conda/3

conda activate /groups/sysops/workshop
python svd_cpu.py
