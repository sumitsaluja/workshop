#!/bin/bash
#SBATCH --job-name=matlab-svd    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --gres=gpu:L40S:1        # number of gpus per node
#SBATCH --partition=gpu          # Queue/Partition

module load matlab/R2023a

matlab -singleCompThread -nodisplay -nosplash -r svd
