#!/bin/bash -l
#SBATCH --job-name=mean-vector  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:L40S:1        # number of gpus per node
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu          # Queue/Partition


# Load the R module
module load R/4.2.1

module load cuda

# Run the R script
Rscript mean_vector.R

