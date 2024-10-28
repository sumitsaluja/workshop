#!/bin/bash -l
#SBATCH --job-name=c++-gpu      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:L40S:1        # number of gpus per node
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --time=00:03:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu          # Queue/Partition

module load cuda

##Compile the cuda script using the nvcc compiler
nvcc matrix_add -o matrix_add.cu

## Run the script
./matrix_add
