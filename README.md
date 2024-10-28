## Welcome to HPC Workshop

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


Access the Notebook

After submitting the job, you'll need to set up port forwarding to access the Jupyter Notebook server.

1. Find the allocated node: Check the output of the job submission to see on which node the Jupyter Notebook is running.

2. SSH into the node: Use the following command to create an SSH tunnel:
```
ssh -N -L $PORT:[HOST]:$PORT [userID]@hpc.c2b2.columbia.edu

```
3. Open your browser: In your web browser, go to http://127.0.0.1:$PORT. You should be able to access your Jupyter Notebook server.




## GPU JOBS

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

Here is the breakdown of what each part does:

Imports:

perf_counter from time for high-resolution timing.

cupy for GPU array operations.

Matrix Generation:

N = 2000 sets the size of the matrix.

X = cp.random.randn(N, N, dtype=cp.float64) creates a 2000x2000 matrix with normally distributed random numbers.


Timing Execution:

The code runs the SVD operation 5 times to measure performance accurately.

cp.linalg.svd(X) computes the SVD of matrix X.

cp.cuda.Device(0).synchronize() ensures that all GPU operations are complete before timing stops.

Results:

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

Breakdown of the Code:

Imports:

perf_counter from the time module for precise timing.

torch for tensor operations and GPU support.

Matrix Setup:

N = 2000 defines the dimensions of the matrix.

cuda0 = torch.device('cuda:0') specifies that computations should occur on the first GPU.

x = torch.randn(N, N, dtype=torch.float64, device=cuda0) generates a 2000x2000 matrix filled with random numbers, stored on the GPU.


SVD Computation:

t0 = perf_counter() starts the timer.
u, s, v = torch.svd(x) computes the SVD of the matrix.
elapsed_time = perf_counter() - t0 calculates the total time taken for the operation.

Results:

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

You can track the job's progress using squeue -u $USER. After the job finishes, check the output using cat slurm-*.out


Submit the job:

```
$ sbatch job.sh
```

