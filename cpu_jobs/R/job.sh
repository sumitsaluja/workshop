#!/bin/bash
#SBATCH --job-name=mean_vectors      # Job name
#SBATCH --output=mean_vectors.out     # Standard output and error log
#SBATCH --ntasks=1                    # Run on a single task
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --time=00:05:00               # Time limit hrs:min:sec
#SBATCH --mem=4G                      # Memory limit

# Load the R module 
module load load R/4.2.1

# Run the R script
Rscript mean_vectors.R
