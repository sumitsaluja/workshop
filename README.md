Welcome to HPC Workshop

Using the CPU/GPUs HPC clusters is easy. Pick one of the applications below to get started. To obtain the materials to run the examples, use these commands:

```
$ ssh <UNI>@hpc.c2b2.columbia.edu
$ cd /group/<PIUNI>_gp/
$ git clone https://github.com/sumitsaluja/workshop.git
```

Juypter
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

