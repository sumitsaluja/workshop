# HPC Workshop

Using the CPU/GPUs HPC clusters is easy. Pick one of the applications below to get started. To obtain the materials to run the examples, use these commands:

```
$ ssh <UNI>@hpc.c2b2.columbia.edu
$ cd /group/<PIUNI>_gp/
$ git clone https://github.com/sumitsaluja/workshop.git
```

## Juypter

This setup allows a user to run Jupyter Lab on a SLURM-managed compute node, facilitating remote access via SSH tunneling. 
The SLURM script handles resource allocation, while the Jupyter script sets up the environment and launches the server, providing clear instructions for accessing the notebook from a local machine.

SLURM Job Script jupyter.slurm

```
#!/bin/bash -l
#SBATCH --job-name=jupyter      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=10G                 # total memory (RAM) per node
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)

module load conda/3

./jupyter.sh

```

The above executes another script jupyter.sh, which contains the commands to set up the Jupyter environment and launch Jupyter Lab

```
export PORT=8888
echo "set up SSH port  forwarding between the compute resource and your local computer (laptop/desktop)"
echo "ssh -N -L $PORT:[HOST]:$PORT [userID]@hpc.c2b2.columbia.edu"
echo "Access the Jupyter Notebook via your web browser on your local computer."
echo "http://127.0.0.1:$PORT/"

module load conda/3
echo " "
echo "============================================================================="
echo "=== jupyterlab.sh Install and run JupyterLab locally"
echo "+++ installing jupyter"
python -m venv .jup
. .jup/bin/activate
python -m pip install pip --upgrade
python -m pip install --upgrade jupyterlab
python -m pip install --upgrade bash_kernel
python -m bash_kernel.install
python -m pip install --upgrade jupyterlab-spellchecker

echo "+++ run jupyter"


jupyter-lab --ip=0.0.0.0 --port=$PORT --no-browser


```


```
jupyter-lab --ip=0.0.0.0 --port=$PORT --no-browser

```
This command starts the Jupyter Lab server, making it accessible from any IP address (0.0.0.0) on the specified port. The --no-browser option prevents Jupyter from trying to open a browser on the compute node, which is typically not possible in HPC environments.


### Access the Notebook

After submitting the job, you'll need to set up port forwarding to access the Jupyter Notebook server.

1. Find the allocated node: Check the output of the job submission to see on which node the Jupyter Notebook is running.

2. SSH into the node: Use the following command to create an SSH tunnel:
```
ssh -N -L $PORT:[HOST]:$PORT [userID]@hpc.c2b2.columbia.edu

```
3. Open your browser: In your web browser, go to http://127.0.0.1:$PORT. You should be able to access your Jupyter Notebook server.




# GPU JOBS

To add a GPU to your Slurm allocation:

```
#SBATCH --gres=gpu:L40S:1     # number of L40S gpus per node
#SBATCH --partition=gpu       #GPU queue
```
## CuPy

[CuPy](https://cupy.chainer.org) is a library that provides an interface similar to NumPy but is designed to leverage NVIDIA GPUs for accelerated computing. It allows users to perform operations on large arrays and matrices efficiently by utilizing the parallel processing power of GPU. You can roughly think of CuPy as NumPy for GPUs


To install CuPy
```
module load conda
conda create --name cupyenv cupy --channel conda-forge

```

In this example we will perform a Singular Value Decomposition (SVD) on a randomly generated matrix using CuPy, which leverages GPU acceleration.

```
from time import perf_counter
import cupy as cp

N = 2000
X = cp.random.randn(N, N, dtype=cp.float64)

trials = 5
times = []
for _ in range(trials):
    t0 = perf_counter()
    u, s, v = cp.linalg.svd(X)
    cp.cuda.Device(0).synchronize()
    times.append(perf_counter() - t0)
print("Execution time: ", min(times))
print("sum(s) = ", s.sum())
print("CuPy version: ", cp.__version__)

```

### Here is the breakdown of what each part does:

### Imports:

perf_counter from time for high-resolution timing.

cupy for GPU array operations.

### Matrix Generation:

N = 2000 sets the size of the matrix.

X = cp.random.randn(N, N, dtype=cp.float64) creates a 2000x2000 matrix with normally distributed random numbers.

### Timing Execution:

The code runs the SVD operation 5 times to measure performance accurately.

cp.linalg.svd(X) computes the SVD of matrix X.

cp.cuda.Device(0).synchronize() ensures that all GPU operations are complete before timing stops.

### Results:

The minimum execution time from the trials is printed.

The sum of the singular values (s) is calculated and displayed.

The CuPy version used is printed.




Below is a sample Slurm script:

```
#!/bin/bash -l
#SBATCH --job-name=cupy-gpu      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:L40S:1        # number of gpus per node
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu          # Queue/Partition

module load conda/3

conda activate /groups/sysops/workshop
python svd.py

```

Submit the job:

```
$ sbatch job.sh
```

You can track the job's progress using squeue -u $USER. After the job finishes, check the output using cat slurm-*.out. What happen if we will double th value of N ?  Will the execution time double? 
There's also a CPU version of the code; let's give that a try.

## PyTorch
[PyTorch](https://pytorch.org) is an open-source machine learning library widely used for deep learning applications. It provides a flexible and dynamic framework for building neural networks, enabling efficient computation on both CPUs and GPUs.

To install PyTorch
```
module load conda
$ conda create --prefix=/users/sysops/tmp/workshop/torch-env torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
$ conda create --name torch-env pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
$ conda activate /users/sysops/tmp/workshop/torch-env

```

In this example we will utilizes PyTorch to perform Singular Value Decomposition (SVD) on a randomly generated matrix, leveraging GPU acceleration. 

```
from time import perf_counter
import torch

N = 2000

cuda0 = torch.device('cuda:0')
x = torch.randn(N, N, dtype=torch.float64, device=cuda0)
t0 = perf_counter()
u, s, v = torch.svd(x)
elapsed_time = perf_counter() - t0

print("Execution time: ", elapsed_time)
print("Result: ", torch.sum(s).cpu().numpy())
print("PyTorch version: ", torch.__version__)

```

### Breakdown of the Code:

### Imports:

perf_counter from the time module for precise timing.

torch for tensor operations and GPU support.

### Matrix Setup:

N = 2000 defines the dimensions of the matrix.

cuda0 = torch.device('cuda:0') specifies that computations should occur on the first GPU.

x = torch.randn(N, N, dtype=torch.float64, device=cuda0) generates a 2000x2000 matrix filled with random numbers, stored on the GPU.


### SVD Computation:

t0 = perf_counter() starts the timer.
u, s, v = torch.svd(x) computes the SVD of the matrix.
elapsed_time = perf_counter() - t0 calculates the total time taken for the operation.

### Results:

The execution time is printed.

The sum of the singular values (s) is computed and transferred back to the CPU for display using .cpu().numpy().

The PyTorch version is printed.



Here is a sample Slurm script:
```
#!/bin/bash -l
#SBATCH --job-name=torch         #  Job name
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:L40S:1        # number of gpus per node
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu          # Queue/Partition

module load conda/3

conda activate /users/sysops/tmp/workshop/torch-env
python svd.py

```
Submit the job:

```
$ sbatch job.sh
```


You can track the job's progress using squeue -u $USER. After the job finishes, check the output using cat slurm-*.out



## MATLAB
MATLAB is available on the cluster.

In this example we will use MATLAB  to perform Singular Value Decomposition (SVD) on a randomly generated matrix, leveraging GPU acceleration.

```
gpu = gpuDevice();
fprintf('Using a %s GPU.\n', gpu.Name);
disp(gpuDevice);

X = gpuArray([1 0 2; -1 5 0; 0 3 -9]);
whos X;
[U,S,V] = svd(X)
fprintf('trace(S): %f\n', trace(S))
quit;

```


### Breakdown of the Code:

### Initialize GPU Device:

gpu = gpuDevice(); retrieves the current GPU device.
fprintf and disp display the name and details of the GPU being used.

### Create a GPU Array:

X is created as a GPU array containing specified values.
whos X; displays information about the variable X.

### Perform SVD:
Computes the SVD of the matrix X, returning matrices U, S, and V.

### Calculate and Display the Trace:

Calculates the trace of matrix S (the sum of its diagonal elements) and prints it.


Here is Slurm Script

```
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


```
Submit the job:

```
$ sbatch job.sh
```


You can track the job's progress using squeue -u $USER. After the job finishes, check the output using cat slurm-*.out


## C++

In this example C++ CUDA code defines a simple kernel that adds the elements of two arrays and demonstrates the use of Unified Memory. 

```
#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}


```
Here's a breakdown of the code:

### Includes:

Includes the necessary headers for input/output and mathematical functions.

### Kernel Function:
This function runs on the GPU and performs element-wise addition of arrays x and y. 

Note that this implementation is not efficient for parallel execution since it runs a loop on a single thread.

### Main Function:
N is set to 2<sup>20</sup>  (1 million elements).

Unified Memory is allocated for the arrays x and y, making them accessible from both the CPU and GPU.

### Initialize Arrays:
The host initializes the arrays x and y with values 1.0 and 2.0, respectively.
### Kernel Launch:

The kernel is launched with one block and one thread. The device is synchronized to ensure the GPU finishes before accessing the results.

### Error Checking:

The maximum error is computed to verify that all elements of y have been correctly updated to 3.0.

### Memory Cleanup:

Frees the allocated Unified Memory for x and y.


### Output

The program will output the maximum error, which should be very close to 0.0 if the addition was successful.


Here is te Slurm Script
```
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


```

This slurm script Compiles the CUDA file named matrix_add.cu using the NVIDIA CUDA Compiler (nvcc), outputting an executable named matrix_add and executes the compiled CUDA application.

Submit the job:

```
$ sbatch job.sh
```




# CPU JOBS

## R
R is already installed in CLUSTER. To use R load the module

```
module load R/4.2.1
```

In this example we will utilizes parallel processing to compute the means of a list of random vectors. To install
```
$ module load load R/4.2.1
$ R
> install.packages(doParallel)
```


Here is the R code

```
mean_vectors.R
library(foreach)
library(doParallel)

# Create a list of 1000 random vectors
vectors <- replicate(1000, rnorm(1000), simplify = FALSE)

# Define a function to compute the mean of a vector
mean_vector <- function(vec) {
  mean(vec)
}

# Compute the means of the vectors using foreach with 4 cores
cl <- makeCluster(4)
registerDoParallel(cl)
start_time <- Sys.time()
means <- foreach(vec = vectors) %dopar% mean_vector(vec)
end_time <- Sys.time()
stopCluster(cl)

# Compute the means of the vectors using a for loop
start_time_serial <- Sys.time()
means_serial <- numeric(length(vectors))
for (i in seq_along(vectors)) {
  means_serial[i] <- mean_vector(vectors[[i]])
}
end_time_serial <- Sys.time()

# Print the execution times
cat("Parallel execution time:", end_time - start_time, "\n")
cat("Serial execution time:", end_time_serial - start_time_serial, "\n")

```

### Code Breakdown:

### Load Libraries:
Loads the necessary libraries for parallel processing.
### Generate Random Vectors:
Creates a list of 1000 random vectors, each containing 1000 normally distributed numbers.
### Define the Mean Function:

Defines a function to compute the mean of a vector.

### Parallel Mean Calculation:

Creates a cluster with 4 cores.

Registers the cluster for parallel processing.

Measures the execution time for computing the means in parallel using foreach.

Stops the cluster after the computation.

### Serial Mean Calculation:

Measures the execution time for computing the means serially using a for loop.

### Print Execution Times:

Outputs the execution times for both parallel and serial computations.


Here i slurm script

```
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



```

This slurm script allocates 4 CPU cores for this task, which matches the parallel processing in R script


